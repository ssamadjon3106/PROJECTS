from fastapi import FastAPI, Request
from app.telegram_api import get_file_url, download_file, send_message
from app.ocr import extract_text_from_image, extract_text_from_pdf
import os

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Bot is running 🚀"}

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()

    if "message" not in data:
        return {"ok": True}

    message = data["message"]
    chat_id = message["chat"]["id"]

    try:
        # 📸 IMAGE
        if "photo" in message:
            file_id = message["photo"][-1]["file_id"]

            file_url = get_file_url(file_id)
            path = "temp.jpg"

            download_file(file_url, path)
            text = extract_text_from_image(path)

            os.remove(path)
            send_message(chat_id, text)

        # 📄 PDF / DOCUMENT
        elif "document" in message:
            file_id = message["document"]["file_id"]
            file_name = message["document"]["file_name"]

            file_url = get_file_url(file_id)

            if file_name.endswith(".pdf"):
                path = "temp.pdf"
                download_file(file_url, path)

                text = extract_text_from_pdf(path)
                os.remove(path)

                send_message(chat_id, text)
            else:
                send_message(chat_id, "Only PDF supported for documents 📄")

    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")

    return {"ok": True}