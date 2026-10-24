# 🪄 Magic Eraser — Deep Learning Image Inpainting & Restoration

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C.svg)](https://pytorch.org/)
[![Author](https://img.shields.io/badge/Author-JIGGISTYLE-181717.svg?logo=github)](https://github.com/JIGGISTYLE)

An end-to-end, deep learning-powered **Image Inpainting & Object Removal** application built entirely
in **Python** with **PyTorch**, a **Streamlit** web UI, and a **FastAPI** inference API.

Upload an image, paint over the objects or damaged regions you want gone, pick one of the pretrained
networks (trained on **Places2** and **CelebA-HQ**), and reconstruct the image in seconds.

---

## 🌟 Key Features

- **🖌️ Interactive Painting Canvas**: A Streamlit drawing surface (brush or rectangle) with adjustable
  brush size, mask preview, and one-click mask clearing.
- **🧠 Multiple Pretrained Backends**:
  - `pt_places2` / `tf_places2`: general scenes and landscapes (Places2 dataset).
  - `pt_celebahq` / `tf_celebahq`: faces and high-quality portrait editing (CelebA-HQ dataset).
  - `pt_*` are the PyTorch-native weights, `tf_*` the TensorFlow-converted ones.
- **⚡ Dual Interface, One Engine**: the Streamlit UI and the FastAPI REST API share the same
  `Inpainter` / `Generator` code — no duplicated inference logic.
- **📊 Multi-stage Visualization**: each run returns both the coarse (stage-1) and the final
  high-fidelity inpainted output, with a PNG download button.
- **🖥️ CPU or GPU**: device is selected automatically (`cuda` when available, otherwise `cpu`).

---

## 🏗️ Architecture & System Overview

```
 ┌────────────────────────┐        upload image + painted mask        ┌─────────────────────────┐
 │  Streamlit UI          │ ────────────────────────────────────────> │                         │
 │  (streamlit_app.py)    │                                           │  Inpainter              │
 └────────────────────────┘ <──────────────────────────────────────── │  (utils/app_utils.py)   │
                                 restored images + download            │            │            │
 ┌────────────────────────┐                                           │            ▼            │
 │  FastAPI API clients   │ ────────────────────────────────────────> │  DeepFill v2 Generator  │
 │  (app.py, /api/inpaint)│ <──────────────────────────────────────── │  (model/networks*.py)   │
 └────────────────────────┘        JSON + /files/*.png                 └─────────────────────────┘
```

---

## 📁 Repository Structure

```
magic-eraser/
├── streamlit_app.py         # Streamlit web UI (upload, paint mask, inpaint, download)
├── app.py                   # FastAPI server & REST endpoints
├── run.bat                  # Launches the Streamlit UI
├── run_api.bat              # Launches the FastAPI API
├── download_files.py        # Fetches the pretrained model weights
├── requirements.txt         # Python dependencies
├── INDEX.md                 # Short codebase index
├── app/
│   ├── models.yaml          # Model name → weights mapping
│   └── files/               # Intermediate outputs served by the API at /files
├── model/                   # Neural networks
│   ├── __init__.py          # load_model(): picks the right Generator for the weights
│   ├── networks.py          # Generator for the PyTorch-native (states_pt_*.pth) weights
│   └── networks_tf.py       # Generator for the TF-converted (states_tf_*.pth) weights
├── utils/
│   ├── app_utils.py         # Inpainter: model loading + request handling
│   └── misc.py              # infer_deepfill() and mask helpers
├── examples/inpaint/        # Sample images + masks (also the Streamlit samples)
└── pretrained/              # Weights directory (.pth state dicts)
```

---

## 🚀 Quick Start & Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional — inference runs on CPU too)

### 1. Clone the Repository

```bash
git clone https://github.com/JIGGISTYLE/magic-eraser.git
cd magic-eraser
```

### 2. Set Up a Virtual Environment & Install Dependencies

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the Pretrained Models

Pulls the checkpoint weights into the `pretrained/` directory:

```bash
python download_files.py
```

---

## 💻 Running the Application

### Streamlit UI (main interface)

```bash
streamlit run streamlit_app.py
# or on Windows: run.bat
```

Opens at **`http://localhost:8501`**. Upload an image (or pick a bundled sample), paint the area to
remove, then click **Run inpainting**.

### FastAPI REST API

```bash
python app.py
# or on Windows: run_api.bat
```

Serves at **`http://127.0.0.1:8000`**, with interactive Swagger docs at **`/docs`**. Output images are
written to `app/files/` and served at `/files/<name>.png`.

---

## 📡 API Reference

### `GET /api/models`

Returns the available pretrained inpainting models and whether each is currently loaded.

### `POST /api/inpaint`

Performs inpainting on an uploaded image with a mask.

- **Form data**:
  - `image`: source image file
  - `mask`: mask image file (white = region to restore)
  - `models`: comma-separated model keys, e.g. `pt_places2` or `pt_places2,tf_places2`
  - `size`: target inference resolution in pixels (default `512`)

```bash
curl -X POST http://127.0.0.1:8000/api/inpaint \
  -F image=@examples/inpaint/case1.png \
  -F mask=@examples/inpaint/case1_mask.png \
  -F models=pt_places2 \
  -F size=512
```

---

## 👤 Author

**Jitesh Kumar Sharma** — [@JIGGISTYLE](https://github.com/JIGGISTYLE)

- GitHub: [github.com/JIGGISTYLE](https://github.com/JIGGISTYLE)
- Repository: [github.com/JIGGISTYLE/magic-eraser](https://github.com/JIGGISTYLE/magic-eraser)
