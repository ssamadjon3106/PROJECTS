import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is missing!")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"