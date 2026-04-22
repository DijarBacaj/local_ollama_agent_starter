from __future__ import annotations

from uuid import uuid4

import gradio as gr

from agent import LocalOllamaAgent
from config import settings

agent = LocalOllamaAgent()


def _normalize_history(history):
    normalized = []
    if not history:
        return normalized

    for item in history:
        if isinstance(item, dict):
            role = str(item.get("role", "assistant"))
            content = "" if item.get("content") is None else str(item.get("content", ""))
            if role in {"user", "assistant", "system"}:
                normalized.append({"role": role, "content": content})

        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg = "" if item[0] is None else str(item[0])
            assistant_msg = "" if item[1] is None else str(item[1])

            if user_msg:
                normalized.append({"role": "user", "content": user_msg})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": assistant_msg})

    return normalized


def respond(message, history, session_id):
    clean_message = (message or "").strip()
    history = _normalize_history(history)
    session_id = session_id or str(uuid4())

    if not clean_message:
        return "", history, session_id

    history.append({"role": "user", "content": clean_message})

    try:
        answer = agent.chat(clean_message, session_id=session_id)
    except Exception as exc:
        answer = (
            "Something went wrong while talking to the local model.\n\n"
            f"Error: {exc}\n\n"
            "Check that Ollama is running and the model name is correct."
        )

    history.append({"role": "assistant", "content": answer})
    return "", history, session_id


def new_chat():
    return [], str(uuid4())


with gr.Blocks(title="Local Ollama Agent Starter") as demo:
    gr.Markdown(
        f"""
# Local Ollama Agent Starter

This is a local AI agent that runs on your PC through Ollama.

**What it can do:**
- chat with a local model
- read and write files in the workspace
- save useful notes in memory
- do simple calculations
- optionally run short Python snippets

**Current model:** `{settings.ollama_model}`
"""
    )

    chatbot = gr.Chatbot(label="Agent", height=520)
    message = gr.Textbox(
        label="Message",
        placeholder="Write a message...",
        lines=3,
    )
    session_id = gr.State(str(uuid4()))

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("New Chat")

    send_btn.click(
        fn=respond,
        inputs=[message, chatbot, session_id],
        outputs=[message, chatbot, session_id],
    )

    message.submit(
        fn=respond,
        inputs=[message, chatbot, session_id],
        outputs=[message, chatbot, session_id],
    )

    clear_btn.click(
        fn=new_chat,
        outputs=[chatbot, session_id],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )