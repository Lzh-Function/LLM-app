import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "llm_chat.server:app",
        host=os.environ.get("LLM_CHAT_HOST", "0.0.0.0"),
        port=int(os.environ.get("LLM_CHAT_PORT", "5070")),
    )
