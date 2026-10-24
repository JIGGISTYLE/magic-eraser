# Codebase Index

This directory (`magic-eraser`) contains a Python image inpainting application built on
DeepFill v2 (gated convolution). It exposes a Streamlit UI and a FastAPI inference API over the
same PyTorch model code.

## Directory Structure

- `streamlit_app.py`: Streamlit web UI (upload, paint mask, run inpainting, download result).
- `app.py`: FastAPI server exposing the same inference as a REST API.
- `run.bat` / `run_api.bat`: one-click launchers for the Streamlit UI / FastAPI API.
- `download_files.py`: downloads the pretrained weights from Google Drive.
- `app/models.yaml`: maps model names (`pt_places2`, `tf_places2`, `pt_celebahq`, `tf_celebahq`) to weight paths.
- `app/files/`: output directory used by the FastAPI API (served at `/files`).
- `model/`: the neural networks.
  - `model/__init__.py`: `load_model()`, picks the right Generator for the given weights.
  - `model/networks.py`: Generator for the PyTorch-native (`states_pt_*.pth`) weights.
  - `model/networks_tf.py`: Generator for the TensorFlow-converted (`states_tf_*.pth`) weights.
- `utils/`: inference plumbing.
  - `utils/app_utils.py`: the `Inpainter` class — loads models and runs requests.
  - `utils/misc.py`: `infer_deepfill` inference loop, config parsing, mask generation.
- `pretrained/`: downloaded model weights (`.pth`).
- `examples/inpaint/`: sample images and masks; `case1-3.png` double as the Streamlit samples.

## Usage Overview
- Download weights with `download_files.py`.
- Launch the UI with `run.bat` (Streamlit) or the API with `run_api.bat` (FastAPI).
- All model/inference logic lives in `utils/` and `model/`; both entry points share it.
