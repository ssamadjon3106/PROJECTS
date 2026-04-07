# app/telegram_api.py
import requests
from app.config import TELEGRAM_API, BOT_TOKEN

def get_file_url(file_id: str):
    res = requests.get(f"{TELEGRAM_API}/getFile?file_id={file_id}").json()
    file_path = res["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

def download_file(file_url: str, save_path: str):
    data = requests.get(file_url).content
    with open(save_path, "wb") as f:
        f.write(data)

def send_message(chat_id: int, text: str):
    if not text:
        text = "No text found 😅"

    # Telegram limit = 4096 chars
    for i in range(0, len(text), 4000):
        chunk = text[i:i+4000]
        requests.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": chat_id,
            "text": chunk
        })