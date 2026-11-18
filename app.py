from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.app_utils import Inpainter

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'app/files'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
inpainter = Inpainter()
inpainter.load_models(ROOT / 'app/models.yaml')

app = FastAPI(title='EraseAI API', docs_url=None, redoc_url=None)
app.mount('/files', StaticFiles(directory=OUTPUT_DIR), 'files')


@app.get('/favicon.svg', include_in_schema=False)
def favicon():
    return FileResponse(ROOT / 'assets/eraseai-icon.svg', media_type='image/svg+xml')


@app.get('/docs', include_in_schema=False)
def swagger_docs():
    return get_swagger_ui_html(openapi_url=app.openapi_url, title='EraseAI API',
                               swagger_favicon_url='/favicon.svg')


@app.get('/redoc', include_in_schema=False)
def redoc_docs():
    return get_redoc_html(openapi_url=app.openapi_url, title='EraseAI API',
                          redoc_favicon_url='/favicon.svg')


@app.get('/api/models')
def get_model_info():
    return {'data': inpainter.get_model_info()}


@app.post('/api/inpaint')
def inpaint(image: UploadFile, mask: UploadFile,
            models: str = Form(), size: int = Form(512, gt=0)):
    # A synchronous route keeps CPU inference off the async event loop.
    try:
        return {'data': inpainter.inpaint(image, mask, models, size)}
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
