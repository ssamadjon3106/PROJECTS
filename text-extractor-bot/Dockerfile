FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]