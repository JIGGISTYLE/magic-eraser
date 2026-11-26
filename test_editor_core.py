import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from editor_core import canvas_geometry, mask_from_canvas, prepare_inputs, run_inpaint


class EditorGeometryTests(unittest.TestCase):
    def test_canvas_fits_without_distorting(self):
        for dimensions in ((512, 365), (365, 512), (2000, 1), (1, 2000)):
            with self.subTest(dimensions=dimensions):
                w, h = canvas_geometry(Image.new('RGB', dimensions), 330)
                self.assertTrue(0 < w <= 330 and h > 0)
                self.assertLessEqual(max(w, h), 640)

    def test_input_padding_preserves_edges(self):
        for dimensions in ((512, 365), (365, 512), (2000, 1), (1, 2000)):
            with self.subTest(dimensions=dimensions):
                image, mask, size = prepare_inputs(
                    Image.new('RGB', dimensions), Image.new('L', dimensions), 512)
                self.assertEqual(image.shape[-1] % 8, 0)
                self.assertEqual(image.shape[-2] % 8, 0)
                self.assertGreater(min(size), 0)

    def test_result_crops_padding_not_photograph(self):
        inpainter = SimpleNamespace(
            check_requested_models=lambda names: None, device='cpu',
            available_models={'test': {'return_vals': ['inpainted']}},
            loaded_models={'test': object()})
        output = np.zeros((368, 512, 3), dtype=np.uint8)
        with patch('editor_core.infer_deepfill', return_value=[output]):
            result = run_inpaint(inpainter, Image.new('RGB', (512, 365)),
                                 Image.new('L', (512, 365)), ['test'], 512)
        self.assertEqual(result[0][1]['inpainted'].size, (512, 365))

    def test_low_resolution_repair_preserves_unmasked_source_pixels(self):
        rng = np.random.default_rng(42)
        original = rng.integers(0, 256, (365, 512, 3), dtype=np.uint8)
        mask = Image.new('L', (512, 365))
        mask.paste(255, (100, 90, 180, 170))
        inpainter = SimpleNamespace(
            check_requested_models=lambda names: None, device='cpu',
            available_models={'test': {'return_vals': ['inpainted', 'stage1']}},
            loaded_models={'test': object()})
        low_res = np.full((96, 128, 3), 200, dtype=np.uint8)
        with patch('editor_core.infer_deepfill', return_value=[low_res, low_res]):
            _, outputs = run_inpaint(inpainter, Image.fromarray(original), mask, ['test'], 128)[0]
        result = np.asarray(outputs['inpainted'])
        outside = np.asarray(mask) == 0
        np.testing.assert_array_equal(result[outside], original[outside])
        self.assertEqual(outputs['inpainted'].size, (512, 365))
        self.assertEqual(outputs['stage1'].size, (128, 91))
        self.assertTrue(np.all(result[100, 110] == 200))

    def test_invalid_input_dimensions_and_size(self):
        image = Image.new('RGB', (32, 24))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            prepare_inputs(image, Image.new('L', (20, 20)), 256)
        with self.assertRaisesRegex(ValueError, 'positive'):
            prepare_inputs(image, Image.new('L', image.size), 0)

    def test_mask_uses_alpha_and_keeps_image_dimensions(self):
        rgba = np.zeros((10, 20, 4), dtype=np.uint8)
        rgba[2:5, 2:5, 3] = 255
        mask = mask_from_canvas(rgba, (100, 50))
        self.assertEqual(mask.size, (100, 50))
        self.assertEqual(mask.getpixel((15, 15)), 255)
        self.assertEqual(mask.getpixel((90, 40)), 0)


if __name__ == '__main__':
    unittest.main()
