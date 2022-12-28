import io
import json

from fastapi import FastAPI, File, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import Response

from segmentation import get_image_from_bytes, get_yolov5, get_yolov5_v2

model = get_yolov5()
modelv2 = get_yolov5_v2()


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


@app.post("/object-to-json-v2", tags=["Image to Json"],  status_code=status.HTTP_200_OK)
async def amount_sheaths_of_wheat(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = modelv2(input_image)
    mydic = {"semillas": 0}
    pred = results.pandas().xyxy[0]
    mydic["vainas_totales"] = len(results.pandas().xyxy[0])

    for index, row in pred.iterrows():
        if row['name'] not in mydic:
            mydic[row['name']] = 1
        else:
            mydic[row['name']] += 1

    for value in mydic:
        if value == "cuatro":
            mydic["semillas"] = mydic["semillas"] + \
                mydic[value] * 4
        elif value == "tres":
            mydic["semillas"] = mydic["semillas"] + \
                mydic[value] * 3
        elif value == "dos":
            mydic["semillas"] = mydic["semillas"] + \
                mydic[value] * 2
        elif value == "uno":
            mydic["semillas"] = mydic["semillas"] + \
                mydic[value] * 1

    return mydic


@app.post("/object-to-json-v1/v2", tags=["Image to Json"],  status_code=status.HTTP_200_OK)
async def amount_sheaths_of_wheat(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    resultsv2 = modelv2(input_image)
    results = model(input_image)
    mydic = {}

    pred = resultsv2.pandas().xyxy[0]

    final_value = {"semillas": 0,
                   "vainas_sobrantes": len(results.pandas().xyxy[0])}

    for index, row in pred.iterrows():
        if row['name'] not in mydic:
            mydic[row['name']] = 1
        else:
            mydic[row['name']] += 1

    for value in mydic:
        if value == "cuatro":
            final_value["vainas_sobrantes"] = final_value["vainas_sobrantes"] - mydic[value]
            final_value["semillas"] = final_value["semillas"] + \
                mydic[value] * 4
        elif value == "tres":
            final_value["vainas_sobrantes"] = final_value["vainas_sobrantes"] - mydic[value]
            final_value["semillas"] = final_value["semillas"] + \
                mydic[value] * 3
        elif value == "dos":
            final_value["vainas_sobrantes"] = final_value["vainas_sobrantes"] - mydic[value]
            final_value["semillas"] = final_value["semillas"] + \
                mydic[value] * 2
        elif value == "uno":
            final_value["vainas_totales"] = final_value["vainas_totales"] - mydic[value]
            final_value["semillas"] = final_value["semillas"] + \
                mydic[value] * 1

    final_value["vainas_totales"] = len(results.pandas().xyxy[0])
    final_value["vainas_totales_inferencia_granos"] = len(
        resultsv2.pandas().xyxy[0])

    mydic["vainas_totales"] = len(resultsv2.pandas().xyxy[0])

    return {"inferencia_vainas": {"vainas_totales": len(results.pandas().xyxy[0])}, "inferencia_granos": mydic, "conteo_final": final_value}


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


@app.post("/object-to-img-v2", tags=["Image to Image"], status_code=status.HTTP_200_OK)
async def return_image_scanned(file: bytes = File(...)):

    input_image = get_image_from_bytes(file)

    results = modelv2(input_image)
    results.render()
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
