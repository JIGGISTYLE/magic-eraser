import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from utils.app_utils import Inpainter

# Route tests do not require downloaded checkpoints.
with patch.object(Inpainter, 'load_models'):
    import app as api


def png(size=(32, 24)):
    buffer = io.BytesIO()
    Image.new('RGB', size).save(buffer, 'PNG')
    return buffer.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.files = {'image': ('image.png', png(), 'image/png'),
                      'mask': ('mask.png', png(), 'image/png')}

    def test_docs_and_favicon(self):
        for path in ('/docs', '/redoc', '/favicon.svg'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn('EraseAI', response.text)

    def test_invalid_size_rejected_before_inference(self):
        with patch.object(api.inpainter, 'inpaint') as run:
            response = self.client.post('/api/inpaint', files=self.files,
                                        data={'models': 'pt_places2', 'size': 0})
        self.assertEqual(response.status_code, 422)
        run.assert_not_called()

    def test_model_error_is_actionable(self):
        with patch.object(api.inpainter, 'inpaint', side_effect=ValueError('Unknown model')):
            response = self.client.post('/api/inpaint', files=self.files,
                                        data={'models': 'missing'})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['detail'], 'Unknown model')

    def test_api_uses_shared_pipeline_and_separate_output_names(self):
        inpainter = Inpainter()
        source = lambda: type('Upload', (), {'file': io.BytesIO(png())})()
        with tempfile.TemporaryDirectory() as folder, \
                patch('utils.app_utils.ROOT', Path(folder)), \
                patch('utils.app_utils.run_inpaint', return_value=[
                    ('test', {'inpainted': Image.new('RGB', (32, 24))})]) as run:
            first = inpainter.inpaint(source(), source(), 'test', 256)
            second = inpainter.inpaint(source(), source(), 'test', 256)
            self.assertEqual(run.call_count, 2)
            self.assertNotEqual(first[0]['output'][0]['file'], second[0]['output'][0]['file'])
            self.assertEqual(len(list((Path(folder) / 'app/files').glob('*.png'))), 2)

    def test_unavailable_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            Inpainter().check_requested_models(['missing'])


if __name__ == '__main__':
    unittest.main()
