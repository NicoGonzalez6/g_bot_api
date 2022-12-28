import torch
from PIL import Image
import io


def get_yolov5():
    # local best.pt
    model = torch.hub.load(
        './yolov5', 'custom', path='./model/best.pt', source='local')  # local repo
    return model


def get_yolov5_v2():
    # local best.pt
    model = torch.hub.load(
        './yolov5', 'custom', path='./model/soja_granos_vainas.pt', source='local')  # local repo
    return model


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image
