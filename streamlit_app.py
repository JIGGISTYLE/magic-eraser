import base64
import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas

from utils.app_utils import Inpainter
from editor_core import canvas_geometry, mask_from_canvas, run_inpaint, png_bytes

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "app" / "models.yaml"
SAMPLES = {
    "Hiker in the mountains (Places2)": ROOT / "examples" / "inpaint" / "case1.png",
    "Wind turbines (Places2)": ROOT / "examples" / "inpaint" / "case2.png",
    "Toucan (Places2)": ROOT / "examples" / "inpaint" / "case3.png",
}

st.set_page_config(
    page_title="EraseAI",
    page_icon=str(ROOT / "assets/eraseai-icon.svg"),
    layout="wide",
)


st.html(f"<style>{(ROOT / 'editor.css').read_text()}</style>")
LOGO_SRC = "data:image/svg+xml;base64," + base64.b64encode((ROOT / "assets/eraseai-logo.svg").read_bytes()).decode()
ICON_SRC = "data:image/svg+xml;base64," + base64.b64encode((ROOT / "assets/eraseai-icon.svg").read_bytes()).decode()


@st.cache_resource(show_spinner="Loading inpainting models…")
def get_inpainter():
    inpainter = Inpainter()
    inpainter.load_models(str(CONFIG_PATH))
    return inpainter


MODEL_LABELS = {
    "pt_places2": "Scenes · Places2",
    "tf_places2": "Scenes · Places2 (converted)",
    "pt_celebahq": "Portraits · CelebA-HQ",
    "tf_celebahq": "Portraits · CelebA-HQ (converted)",
}


def choose_sample(name):
    st.session_state.sample = name


has_source = bool(st.session_state.get("photo") or st.session_state.get("sample", "None") != "None")
inpainter = get_inpainter() if has_source else None

with st.sidebar:
    st.html(f'<div class="editor-brand"><img src="{LOGO_SRC}" alt="EraseAI" width="210" height="49"></div>')
    if has_source:
        st.subheader("Selection tools")
        tool = st.radio("Tool", ["Brush", "Rectangle"], horizontal=True)
        brush_size = st.slider("Brush size", 2, 120, 30, format="%d px")
        stroke_color = st.color_picker("Mask color", "#ffffff")
        st.caption("Cover the object and a little of its edges for a cleaner repair.")

        st.divider()
        st.subheader("Restoration")
        available = list(inpainter.available_models.keys())
        model_names = st.multiselect(
            "Model",
            available,
            default=["pt_places2"] if "pt_places2" in available else available[:1],
            format_func=lambda name: MODEL_LABELS.get(name, name),
            help="Scenes models suit landscapes and everyday photos. Portrait models suit faces. Select multiple models to compare results.",
        )
        with st.expander("Model details"):
            st.caption(f"Processing device: {inpainter.device}")
            for name in available:
                status = "Loaded" if name in inpainter.loaded_models else "Loads on use"
                st.caption(f"{name} · {status}")
        st.html('<p class="editor-note">Images are processed by this local app.<br>Export your finished image as a PNG.</p>')

st.html(f'<div class="main-brand"><h1><img src="{ICON_SRC}" alt="EraseAI" width="48" height="48"></h1></div>')
st.caption("Paint over an object. Restore the space it leaves behind.")

source_panel = (st.expander("Change photograph", expanded=False, key="source-edit")
                if has_source else st.container(key="landing"))
with source_panel:
    if not has_source:
        preview, chooser = st.columns([1.65, 1], gap="large")
        with preview, st.container(key="landing-preview"):
            st.subheader("See what a small edit can do")
            original, restored = st.tabs(["Original", "Erased"])
            with original:
                st.image(Image.open(SAMPLES["Wind turbines (Places2)"]), width="stretch")
            with restored:
                st.image(Image.open(ROOT / "examples/inpaint/case2_restored.png"), width="stretch")
            st.caption("The distant turbine, removed with this app. Switch views to compare.")
        upload_panel = chooser.container(border=True, key="upload-panel")
    else:
        upload_panel = st.container()
    with upload_panel:
        if not has_source:
            st.subheader("Make room for what matters.")
            st.caption("Open your photograph to remove an object or repair a detail.")
        upload = st.file_uploader(
            "Your photograph", type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="photo", label_visibility="collapsed",
        )
        if not has_source:
            with st.container(key="landing-guide"):
                st.html('''<ol class="upload-steps">
                    <li><span aria-hidden="true">01</span><div><strong>Open</strong><p>Upload a photograph or choose a sample below.</p></div></li>
                    <li><span aria-hidden="true">02</span><div><strong>Select</strong><p>Paint over the object with the brush or rectangle.</p></div></li>
                    <li><span aria-hidden="true">03</span><div><strong>Restore</strong><p>Review the result and download a PNG.</p></div></li>
                </ol><p class="upload-privacy">Processed locally on this computer.</p>''')
    if not has_source:
        sample_panel = chooser.container(key="sample-picker")
    if has_source:
        sample = st.selectbox("Try a sample", ["None"] + list(SAMPLES), key="sample")
    else:
        sample = st.session_state.get("sample", "None")


if upload is not None and not has_source:
    st.rerun()

if upload is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("This file could not be opened as a photograph. Try another PNG, JPG or WebP image.")
        st.stop()
    image_id = hashlib.sha256(upload.getvalue()).hexdigest()
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
    if has_source:
        st.rerun()
    with sample_panel:
        st.subheader("Try a sample")
        with st.container(key="sample-gallery"):
            for col, (name, path) in zip(st.columns(3, gap="medium"), SAMPLES.items()):
                with col:
                    st.image(Image.open(path), width="stretch")
                    st.button(name.split(" (")[0].replace("Hiker in the mountains", "Mountain hiker"), key=f"sample-{path.stem}",
                              on_click=choose_sample, args=(name,), width="stretch")
    st.stop()

for key, default in (("canvas_token", 0), ("inpaint_key", None), ("results", None)):
    st.session_state.setdefault(key, default)

photo_info, inference_control = st.columns([1, 1.6], gap="medium")
with photo_info:
    st.caption(f"{image_stem if sample == 'None' or upload else sample.split(' (')[0]} · {image.width} × {image.height} px")
with inference_control:
    max_size = st.select_slider(
        "Inference size", [256, 384, 512, 640, 768], value=512,
        format_func=lambda size: {
            256: "256 px · Fastest", 384: "384 px · Faster", 512: "512 px · Balanced",
            640: "640 px · Slower", 768: "768 px · Most detail",
        }[size],
        help="Sets the longest image edge processed by the model. Smaller sizes are faster; larger sizes take longer and may retain more detail. A larger size does not guarantee a better repair.",
    )
    st.caption("Smaller: faster preview. Larger: slower processing, more detail potential.")

# Measure the editor, not the whole window, so the canvas also fits beside the sidebar.
measure_editor = st.components.v2.component(
    "editor_width", html='<div style="height:0;width:100%"></div>',
    js="""export default function({parentElement, setStateValue}) {
        const element = parentElement.querySelector('div');
        let previous = 0;
        let timer;
        const observer = new ResizeObserver(entries => {
            clearTimeout(timer);
            timer = setTimeout(() => {
                const width = Math.floor(entries[0].contentRect.width);
                if (width > 0 && width !== previous) {
                    previous = width;
                    setStateValue('pixels', width);
                }
            }, 150);
        });
        observer.observe(element);
        return () => { clearTimeout(timer); observer.disconnect(); };
    }""",
)
width_state = measure_editor(key="editor-width", default={"pixels": 640},
                             on_pixels_change=lambda: None, height=0, width="stretch")
canvas_w, canvas_h = canvas_geometry(image, max(1, (width_state.pixels or 640) - 26))
drawing_mode = "freedraw" if tool == "Brush" else "rect"
fill_color = "rgba(255,255,255,1)" if tool == "Rectangle" else stroke_color

with st.container(key="editor-toolbar"):
    instruction, clear, restore = st.columns([3, 1, 1.3], gap="small")
    instruction.markdown("**Paint the area to remove**")
    if clear.button("Clear selection", width="stretch"):
        st.session_state.canvas_token += 1
        st.session_state.results = None
        st.rerun()
    run_clicked = restore.button("Restore image", type="primary", width="stretch")

canvas_base = f"canvas-{image_id}-{st.session_state.canvas_token}"
canvas_key = f"{canvas_base}-{canvas_w}"
previous = st.session_state.get("canvas_snapshot")
if previous and previous["base"] == canvas_base and previous["width"] != canvas_w:
    drawing = copy.deepcopy(previous["drawing"])
    scale = canvas_w / previous["width"]
    for obj in drawing.get("objects", []):
        for field in ("left", "top"):
            obj[field] = obj.get(field, 0) * scale
        for field in ("scaleX", "scaleY"):
            obj[field] = obj.get(field, 1) * scale
    st.session_state["canvas_initial"] = {"key": canvas_key, "drawing": drawing}
initial = st.session_state.get("canvas_initial", {})
with st.container(horizontal_alignment="center"):
    with st.container(key="canvas-stage", width=canvas_w + 26, horizontal_alignment="center"):
        canvas_result = st_canvas(
            fill_color=fill_color, stroke_width=brush_size, stroke_color=stroke_color,
            background_image=image, background_image_fit="contain", update_streamlit=True,
            height=canvas_h, width=canvas_w, drawing_mode=drawing_mode,
            return_image_data=True, key=canvas_key,
            initial_drawing=initial.get("drawing") if initial.get("key") == canvas_key else None,
        )

if canvas_result.json_data is not None:
    st.session_state.canvas_snapshot = {
        "base": canvas_base, "width": canvas_w, "drawing": canvas_result.json_data,
    }

canvas_pixels = canvas_result.image_data
previous_mask = st.session_state.get("canvas_mask")
if (canvas_pixels is None and previous_mask and previous_mask["base"] == canvas_base
        and (canvas_result.json_data or {}).get("objects")):
    # A resized canvas draws initial objects before it emits fresh pixel data.
    mask = previous_mask["image"]
else:
    mask = mask_from_canvas(canvas_pixels, image.size)
if canvas_pixels is not None:
    st.session_state.canvas_mask = {"base": canvas_base, "image": mask}
drawn_pixels = int((np.asarray(mask) > 0).sum())
request_key = (image_id, hashlib.sha256(mask.tobytes()).hexdigest(), tuple(model_names), max_size)
st.caption(f"Selection: {drawn_pixels / (image.width * image.height):.1%} of the photograph · Brush and rectangle marks are rebuilt from the surrounding image.")

with st.expander("Preview mask", expanded=False):
    cols = st.columns(2)
    cols[0].image(image, caption="Input", width=min(image.width, 480))
    cols[1].image(mask, caption=f"Mask ({drawn_pixels} px)", width=min(image.width, 480))

if run_clicked:
    if not model_names:
        st.warning("Select at least one model in the sidebar.")
    elif drawn_pixels == 0:
        st.warning("Paint over something first — the mask is empty.")
    else:
        started = time.perf_counter()
        try:
            with st.spinner("Restoring selection…"):
                st.session_state.results = run_inpaint(
                    inpainter, image, mask, model_names, max_size
                )
            st.session_state.inpaint_key = request_key
            st.session_state.elapsed = time.perf_counter() - started
        except (RuntimeError, ValueError, OSError) as error:
            st.error(f"Restoration failed: {error}. Try a smaller output size.")

st.divider()
st.subheader("Result")

if not st.session_state.results:
    st.caption("Your restored image will appear here. Mark an area, then choose **Restore image**.")
elif st.session_state.inpaint_key != request_key:
    st.info("The image, selection, or settings changed. Restore again to update the result.")
else:
    for name, outputs in st.session_state.results:
        st.markdown(f"**{MODEL_LABELS.get(name, name)}**")
        inpainted = outputs.get("inpainted")
        if inpainted is not None:
            before, after = st.columns(2, gap="medium")
            before.image(image.resize(inpainted.size, Image.Resampling.LANCZOS),
                         caption="Original", width="stretch")
            after.image(inpainted, caption=f"Restored · {inpainted.width} × {inpainted.height} px",
                        width="stretch")
            st.caption(f"Restored in {st.session_state.get('elapsed', 0):.1f} s · {inpainter.device} · PNG export keeps these dimensions.")
            st.download_button(
                "Download PNG",
                png_bytes(inpainted),
                file_name=f"{image_stem}_{name}_inpainted.png",
                mime="image/png",
                key=f"download-{name}",
            )
        stage1 = outputs.get("stage1")
        if stage1 is not None:
            with st.expander("Show coarse (stage-1) result"):
                st.image(stage1, width=min(stage1.width, canvas_w))
        st.divider()
