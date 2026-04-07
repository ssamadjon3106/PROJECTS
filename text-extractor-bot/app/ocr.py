import easyocr
import numpy as np
from PIL import Image
from io import BytesIO

# load once
reader = easyocr.Reader(['en', 'ru'])


def run_ocr(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)

    result = reader.readtext(img_np)

    return " ".join([item[1] for item in result])