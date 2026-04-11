from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zai_proxy.client import ZaiClient
from zai_proxy.config import Settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Delete z.ai chats from the current account.")
    parser.add_argument("--type", default="default", help="Chat type to delete. Default: default")
    parser.add_argument(
        "--keep",
        type=int,
        default=0,
        help="Keep the newest N chats and delete the rest. Default: 0",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    client = ZaiClient(settings)
    deleted = 0
    kept = 0

    try:
        page = 1
        chats: list[dict] = []
        while True:
            batch = await client.list_chats(page=page, chat_type=args.type)
            if not batch:
                break
            chats.extend(batch)
            page += 1

        for index, chat in enumerate(chats):
            if index < args.keep:
                kept += 1
                continue

            chat_id = str(chat.get("id") or "").strip()
            if not chat_id:
                continue

            ok = await client.delete_chat(chat_id)
            if ok:
                deleted += 1
                print(f"deleted {chat_id} {chat.get('title', '')}".strip())

        print(f"done deleted={deleted} kept={kept} total={len(chats)}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
