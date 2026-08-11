from __future__ import annotations

from .config import ENV_PATH, QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from .llm_client import QwenClient


def mask_key(value: str) -> str:
    if not value:
        return "未读取到"
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-4:]}"


def main() -> None:
    client = QwenClient()
    print(f".env path: {ENV_PATH}")
    print(f".env exists: {ENV_PATH.exists()}")
    print(f"QWEN_API_KEY configured: {bool(QWEN_API_KEY)}")
    print(f"QWEN_API_KEY masked: {mask_key(QWEN_API_KEY)}")
    print(f"QWEN_BASE_URL: {QWEN_BASE_URL}")
    print(f"QWEN_MODEL: {QWEN_MODEL}")
    print(f"chat_completions_url: {client.chat_completions_url}")


if __name__ == "__main__":
    main()
