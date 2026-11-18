"""Image preparation and inference shared by the editor and its checks."""
import io
import numpy as np
from PIL import Image
from torch.nn.functional import pad
from torchvision.transforms import ToTensor
from utils.misc import infer_deepfill

MAX_CANVAS_PX = 640


def canvas_geometry(image, available_width=MAX_CANVAS_PX):
    w, h = image.size
    scale = min(MAX_CANVAS_PX / max(w, h), available_width / w)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


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
    size = (max_size, max(1, int(round(scale * mh)))) if mw > mh else (max(1, int(round(scale * mw))), max_size)
    mask = mask.resize(size, Image.NEAREST)
    image = image.resize(size, Image.BICUBIC)
    image_t, mask_t = ToTensor()(image), ToTensor()(mask)
    # DeepFill crops nonmultiples of eight; pad first to preserve every edge.
    padding = (0, (-size[0]) % 8, 0, (-size[1]) % 8)
    return pad(image_t, padding, mode="replicate"), pad(mask_t, padding), size


def run_inpaint(inpainter, image, mask, model_names, max_size):
    inpainter.check_requested_models(model_names)
    image_t, mask_t, output_size = prepare_inputs(image, mask, max_size)
    results = []
    for name in model_names:
        return_vals = inpainter.available_models[name]["return_vals"]
        outputs = infer_deepfill(
            inpainter.loaded_models[name],
            image_t.to(inpainter.device),
            mask_t.to(inpainter.device),
            return_vals=return_vals,
        )
        results.append((name, dict(zip(return_vals, [Image.fromarray(o).crop((0, 0, *output_size)) for o in outputs]))))
    return results


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
