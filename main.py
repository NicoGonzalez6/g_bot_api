import io
import json

from fastapi import FastAPI, File, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import Response

from segmentation import get_image_from_bytes, get_yolov5

model = get_yolov5()

app = FastAPI(
    title="Custom G Bot Api",
    version="0.0.1",
)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/object-to-json", tags=["Image to Json"],  status_code=status.HTTP_200_OK)
async def amount_sheaths_of_wheat(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    return {"vainas": len(results.pandas().xyxy[0])}


@app.post("/object-to-img", tags=["Image to Image"], status_code=status.HTTP_200_OK)
async def return_image_scanned(file: bytes = File(...)):

    input_image = get_image_from_bytes(file)

    results = model(input_image)
    results.render()
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
