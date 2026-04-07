import httpx
from app.config import TELEGRAM_API, BOT_TOKEN


async def get_file_url(file_id: str):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{TELEGRAM_API}/getFile",
            params={"file_id": file_id}
        )
        data = res.json()
        file_path = data["result"]["file_path"]
        return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"


async def download_image(url: str):
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        return res.content


async def send_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:4000],
            }
        )