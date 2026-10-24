import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import ToTensor

from utils.app_utils import Inpainter
from utils.misc import infer_deepfill

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "app" / "models.yaml"
SAMPLES = {
    "Man on a boat (Places2)": ROOT / "examples" / "inpaint" / "case1.png",
    "Statue in a park (Places2)": ROOT / "examples" / "inpaint" / "case2.png",
    "Person in a street (Places2)": ROOT / "examples" / "inpaint" / "case3.png",
}
MAX_CANVAS_PX = 640

st.set_page_config(
    page_title="Magic Eraser · DeepFill v2",
    page_icon="🪄",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading inpainting models…")
def get_inpainter():
    inpainter = Inpainter()
    inpainter.load_models(str(CONFIG_PATH))
    return inpainter


def canvas_geometry(image):
    w, h = image.size
    scale = MAX_CANVAS_PX / max(w, h)
    return int(round(w * scale)), int(round(h * scale))


def mask_from_canvas(image_data, target_size):
    if image_data is None:
        return Image.new("L", target_size, 0)
    alpha = np.asarray(image_data)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 3]
    mask = Image.fromarray((alpha > 0).astype(np.uint8) * 255, mode="L")
    return mask.resize(target_size, Image.NEAREST)


def prepare_inputs(image, mask, max_size):
    mw, mh = mask.size
    scale = max_size / max(mw, mh)
    size = (max_size, int(round(scale * mh))) if mw > mh else (int(round(scale * mw)), max_size)
    mask = mask.resize(size, Image.NEAREST)
    image = image.resize(size, Image.BICUBIC)
    return ToTensor()(image), ToTensor()(mask)


def run_inpaint(inpainter, image, mask, model_names, max_size):
    inpainter.check_requested_models(model_names)
    image_t, mask_t = prepare_inputs(image, mask, max_size)
    results = []
    for name in model_names:
        return_vals = inpainter.available_models[name]["return_vals"]
        outputs = infer_deepfill(
            inpainter.loaded_models[name],
            image_t.to(inpainter.device),
            mask_t.to(inpainter.device),
            return_vals=return_vals,
        )
        results.append((name, dict(zip(return_vals, [Image.fromarray(o) for o in outputs]))))
    return results


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("🪄 Magic Eraser")
st.caption(
    "Remove objects and repair images with DeepFill v2 (gated convolution). "
    "Upload a photo, paint over what you want gone, then run inpainting."
)

inpainter = get_inpainter()

with st.sidebar:
    st.header("Model")
    available = list(inpainter.available_models.keys())
    model_names = st.multiselect(
        "Inpainting models",
        available,
        default=["pt_places2"],
        help="pt_* are PyTorch-native weights, tf_* are the TensorFlow-converted "
        "weights. places2 is best for scenes, celebahq for faces.",
    )
    max_size = st.select_slider(
        "Inference size (px)", [256, 384, 512, 640, 768], value=512
    )

    st.header("Brush")
    tool = st.radio("Tool", ["Brush", "Rectangle"], horizontal=True)
    brush_size = st.slider("Brush size", 2, 120, 30)
    stroke_color = st.color_picker("Mask color", "#ffffff")

    st.divider()
    st.caption(f"Device: `{inpainter.device}`")
    for name in available:
        loaded = "✅" if name in inpainter.loaded_models else "·"
        st.caption(f"{loaded} `{name}`")

with st.container():
    left, right = st.columns([2, 1], gap="large")
    with left:
        upload = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )
    with right:
        sample = st.selectbox("…or try a sample", ["None"] + list(SAMPLES))
        st.write("")
        st.write("")

if upload is not None:
    image = Image.open(upload).convert("RGB")
    image_id = f"{upload.name}-{upload.size}"
    image_stem = Path(upload.name).stem
elif sample != "None":
    image = Image.open(SAMPLES[sample]).convert("RGB")
    image_id = f"sample-{sample}"
    image_stem = "sample"
else:
    image = None
    image_id = "none"
    image_stem = "image"

if image is None:
    st.info("Upload an image or pick a sample to begin.")
    st.stop()

for key, default in (("canvas_token", 0), ("inpaint_key", None), ("results", None)):
    st.session_state.setdefault(key, default)

canvas_w, canvas_h = canvas_geometry(image)
drawing_mode = "freedraw" if tool == "Brush" else "rect"
fill_color = "rgba(255,255,255,1)" if tool == "Rectangle" else stroke_color

st.subheader("Paint the area to remove")
canvas_key = f"canvas-{image_id}-{st.session_state.canvas_token}"

col_canvas, col_controls = st.columns([3, 1])
with col_canvas:
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=brush_size,
        stroke_color=stroke_color,
        background_image=image,
        background_image_fit="contain",
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode=drawing_mode,
        return_image_data=True,
        key=canvas_key,
    )
with col_controls:
    st.write("")
    if st.button("🧹 Clear mask", width="stretch"):
        st.session_state.canvas_token += 1
        st.session_state.results = None
        st.rerun()
    run_clicked = st.button(
        "✨ Run inpainting", type="primary", width="stretch"
    )
    st.caption("Painting is stored as a white mask; everything you cover gets rebuilt.")

mask = mask_from_canvas(canvas_result.image_data, image.size)
drawn_pixels = int((np.asarray(mask) > 0).sum())

with st.expander("Preview mask", expanded=False):
    cols = st.columns(2)
    cols[0].image(image, caption="Input", width="stretch")
    cols[1].image(mask, caption=f"Mask ({drawn_pixels} px)", width="stretch")

if run_clicked:
    if not model_names:
        st.warning("Select at least one model in the sidebar.")
    elif drawn_pixels == 0:
        st.warning("Paint over something first — the mask is empty.")
    else:
        with st.spinner("Running inpainting…"):
            st.session_state.results = run_inpaint(
                inpainter, image, mask, model_names, max_size
            )
        st.session_state.inpaint_key = f"{image_id}-{st.session_state.canvas_token}"

st.divider()
st.subheader("Result")

if not st.session_state.results:
    st.caption("No result yet — paint a mask and hit **Run inpainting**.")
elif st.session_state.inpaint_key != f"{image_id}-{st.session_state.canvas_token}":
    st.caption("Image or mask changed — run inpainting again to refresh the result.")
else:
    for name, outputs in st.session_state.results:
        st.markdown(f"**{name}**")
        inpainted = outputs.get("inpainted")
        if inpainted is not None:
            st.image(
                inpainted,
                caption=f"{inpainted.size[0]}×{inpainted.size[1]} · inpainted",
                width="stretch",
            )
            st.download_button(
                "⬇️ Download result",
                png_bytes(inpainted),
                file_name=f"{image_stem}_{name}_inpainted.png",
                mime="image/png",
                key=f"download-{name}",
            )
        stage1 = outputs.get("stage1")
        if stage1 is not None:
            with st.expander("Show coarse (stage-1) result"):
                st.image(stage1, width="stretch")
        st.divider()
