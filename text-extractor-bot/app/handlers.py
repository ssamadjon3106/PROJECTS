import asyncio
from app.telegram import get_file_url, download_image, send_message
from app.ocr import run_ocr
from app.utils import clean_text


async def process_update(data: dict):
    try:
        message = data.get("message")
        if not message:
            return

        if message.get("from", {}).get("is_bot"):
            return

        chat_id = message["chat"]["id"]

        if "photo" not in message:
            return

        file_id = message["photo"][-1]["file_id"]

        await send_message(chat_id, "⏳ Processing image...")

        file_url = await get_file_url(file_id)
        image_bytes = await download_image(file_url)

        # run OCR in thread
        loop = asyncio.get_event_loop()
        raw_text = await loop.run_in_executor(None, run_ocr, image_bytes)

        cleaned = clean_text(raw_text)

        if not cleaned:
            cleaned = "⚠️ No text detected."

        response = f"📄 Extracted Text:\n\n{cleaned}\n\n✅ Done"

        await send_message(chat_id, response)

    except Exception as e:
        print("ERROR:", e)