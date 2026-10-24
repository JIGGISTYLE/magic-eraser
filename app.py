import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.staticfiles import StaticFiles

from utils.app_utils import Inpainter

inpainter = Inpainter()
inpainter.load_models('app/models.yaml')

app = FastAPI(title='DeepFill v2 Inpainting API')

app.mount('/files', StaticFiles(directory='app/files'), 'files')


@app.get('/api/models')
async def get_model_info():
    return {'data': inpainter.get_model_info()}


@app.post('/api/inpaint')
async def inpaint(image: UploadFile,
                  mask: UploadFile,
                  models: str = Form(),
                  size: int = Form(512)):
    response_data = inpainter.inpaint(image, mask, models, size)
    return {'data': response_data}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
