from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    temperature: float
    max_tool_steps: int
    max_file_chars: int
    allow_code_execution: bool
    gradio_server_name: str
    gradio_server_port: int
    workspace_dir: Path
    memory_db: Path


def load_settings() -> Settings:
    workspace_dir = Path(os.getenv("WORKSPACE_DIR", str(BASE_DIR / "workspace"))).resolve()
    memory_db = Path(os.getenv("MEMORY_DB", str(BASE_DIR / "memory" / "agent_memory.db"))).resolve()

    workspace_dir.mkdir(parents=True, exist_ok=True)
    memory_db.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.2")),
        max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "5")),
        max_file_chars=int(os.getenv("MAX_FILE_CHARS", "12000")),
        allow_code_execution=_as_bool(os.getenv("ALLOW_CODE_EXECUTION", "false")),
        gradio_server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        gradio_server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        workspace_dir=workspace_dir,
        memory_db=memory_db,
    )


settings = load_settings()
