import cv2
import pytesseract
from pdf2image import convert_from_path
import os

def preprocess(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text_from_image(path):
    processed = preprocess(path)
    return pytesseract.image_to_string(processed, lang="eng+rus").strip()

def extract_text_from_pdf(path):
    pages = convert_from_path(path)
    full_text = ""

    for i, page in enumerate(pages):
        temp_img = f"page_{i}.jpg"
        page.save(temp_img, "JPEG")
        full_text += extract_text_from_image(temp_img) + "\n"

        os.remove(temp_img)

    return full_text.strip()