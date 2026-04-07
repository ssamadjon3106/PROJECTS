from fastapi import FastAPI, Request
import asyncio
from app.handlers import process_update

app = FastAPI()

processed_updates = set()


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    update_id = data.get("update_id")

    # prevent duplicates
    if update_id in processed_updates:
        return {"ok": True}

    processed_updates.add(update_id)

    # background task
    asyncio.create_task(process_update(data))

    return {"ok": True}


@app.get("/")
async def root():
    return {"status": "Bot running 🚀"}