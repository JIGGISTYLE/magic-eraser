from pathlib import Path
from uuid import uuid4

import torch
import yaml
from PIL import Image, ImageOps

from editor_core import run_inpaint
from model import load_model

ROOT = Path(__file__).resolve().parent.parent


def _load_models(config_path, device='cuda'):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    config = {name: cfg for name, cfg in config.items()
              if (ROOT / cfg['path']).is_file()}
    loaded_models = {}
    for name, cfg in config.items():
        if cfg['load_at_startup']:
            loaded_models[name] = load_model(ROOT / cfg['path'], device)
        cfg['is_loaded'] = name in loaded_models
    return config, loaded_models


class Inpainter:
    def __init__(self, device=None):
        self.available_models = {}
        self.loaded_models = {}
        self.host = ""
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def load_models(self, config_path):
        self.available_models, self.loaded_models = _load_models(config_path, self.device)

    def get_model_info(self):
        return [dict(cfg, name=name, type='df')
                for name, cfg in self.available_models.items()]

    def check_requested_models(self, models):
        for name in models:
            if name not in self.available_models:
                raise ValueError(f"Unknown or unavailable model: {name}")
            if name not in self.loaded_models:
                path = ROOT / self.available_models[name]['path']
                model = load_model(path, self.device)
                if model is None:
                    raise ValueError(f"Model weights are missing: {name}")
                self.loaded_models[name] = model
                self.available_models[name]['is_loaded'] = True

    def inpaint(self, image, mask, models, max_size=512):
        req_models = [name.strip() for name in models.split(',') if name.strip()]
        if not req_models:
            raise ValueError("Select at least one model")
        with Image.open(image.file) as source:
            image_pil = ImageOps.exif_transpose(source).convert('RGB')
        with Image.open(mask.file) as source:
            mask_pil = ImageOps.exif_transpose(source).convert('L')

        results = run_inpaint(self, image_pil, mask_pil, req_models, max_size)
        output_dir = ROOT / 'app/files'
        output_dir.mkdir(parents=True, exist_ok=True)
        # Separate requests must not overwrite another user's downloadable result.
        request_id = uuid4().hex
        response_data = []
        for idx_model, (name, outputs) in enumerate(results):
            output_data = []
            for idx_out, (output_name, output) in enumerate(outputs.items()):
                filename = f'{request_id}_{idx_model}_{idx_out}.png'
                output.save(output_dir / filename)
                output_data.append({'name': output_name, 'file': f'{self.host}/files/{filename}'})
            response_data.append({'name': name, 'output': output_data})
        return response_data
