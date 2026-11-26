import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from editor_core import run_inpaint
from model import load_model
from model.deepfill_lite import DeepFillLite, DeepFillLiteGenerator


class ModelOptionsTests(unittest.TestCase):
    def test_lite_adapter_matches_training_model(self):
        network = DeepFillLite(8).eval()
        checkpoint = {'channels': 8, 'state_dict': network.state_dict()}
        adapter = DeepFillLiteGenerator(checkpoint).eval()
        image = torch.rand(1, 3, 32, 32)*2-1
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1
        inputs = torch.cat([image*(1-mask), torch.ones_like(mask), mask], 1)
        with torch.inference_mode():
            expected = network(image, mask)
            actual = adapter(inputs, mask)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)

    def test_export_loads_with_correct_architecture(self):
        import tempfile
        checkpoint = {'architecture': 'DeepFillLite-two-stage-no-contextual-attention',
                      'channels': 8, 'state_dict': DeepFillLite(8).state_dict()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'weights.pt'
            torch.save(checkpoint, path)
            model = load_model(path, 'cpu')
        self.assertIsInstance(model, DeepFillLiteGenerator)
        self.assertFalse(model.training)

    def test_lite_rejects_unvalidated_size(self):
        inpainter = SimpleNamespace(check_requested_models=lambda names: None,
                                   available_models={'deepfill_lite': {'inference_size': 128}})
        with patch('editor_core.infer_deepfill') as infer:
            with self.assertRaisesRegex(ValueError, 'requires inference size 128'):
                run_inpaint(inpainter, Image.new('RGB', (32, 32)),
                            Image.new('L', (32, 32)), ['deepfill_lite'], 512)
        infer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
