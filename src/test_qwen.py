from __future__ import annotations

import json

import requests

from .config import ENV_PATH, QWEN_API_KEY, QWEN_MODEL
from .llm_client import QwenClient


def mask_key(value: str) -> str:
    if not value:
        return "未读取到"
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-4:]}"


def main() -> None:
    client = QwenClient()
    chat_completions_url = f"{client.base_url}/chat/completions"
    print(f".env path: {ENV_PATH}")
    print(f".env exists: {ENV_PATH.exists()}")
    print(f"key configured: {bool(QWEN_API_KEY)}")
    print(f"key masked: {mask_key(QWEN_API_KEY)}")
    print(f"model: {QWEN_MODEL}")
    print(f"url: {chat_completions_url}")

    if not QWEN_API_KEY:
        print("ERROR: 没有读到 QWEN_API_KEY，请检查 .env 变量名。")
        return

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "user", "content": "请只回复：ok"}
        ],
        "temperature": 0,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            chat_completions_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        print(f"status_code: {response.status_code}")
        try:
            data = response.json()
            print("response_json:")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        except ValueError:
            print("response_text:")
            print(response.text[:2000])

        if response.ok:
            print("Qwen API test: OK")
        else:
            print("Qwen API test: FAILED")
    except Exception as exc:
        print(f"request_exception: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
