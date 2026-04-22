from __future__ import annotations

import json
import re
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from config import settings
from memory import MemoryStore
from tools import ToolRegistry


class LocalOllamaAgent:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
        )
        self.memory = MemoryStore(settings.memory_db)
        self.tools = ToolRegistry(self.memory)

    def _system_prompt(self) -> str:
        return f"""You are a local Python agent running on the user's computer.

You can think step by step, but you MUST respond with exactly one JSON object and nothing else.

Available tools:
{self.tools.describe_tools_for_prompt()}

Current workspace folder:
{settings.workspace_dir}

Response schema:
1) When you need a tool:
{{
  "type": "tool",
  "tool": "<tool_name>",
  "arguments": {{ ... }},
  "reason": "short reason"
}}

2) When you are ready to answer the user:
{{
  "type": "final",
  "answer": "your helpful final answer",
  "remember": ["optional durable fact 1", "optional durable fact 2"]
}}

Rules:
- Never invent tool results.
- Keep arguments valid JSON.
- Use tools when they would improve accuracy or help inspect files.
- Only save durable notes when it is genuinely useful for future chats.
- Do not wrap JSON in markdown.
"""

    def _extract_json(self, text: str) -> dict[str, Any]:
        raw = text.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1))

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])

        raise ValueError("Model did not return valid JSON.")

    def _append_message(
        self,
        messages: list[ChatCompletionMessageParam],
        role: str,
        content: str,
    ) -> None:
        safe_role = role if role in {"system", "user", "assistant"} else "user"
        messages.append(
            cast(ChatCompletionMessageParam, {"role": safe_role, "content": content})
        )

    def _build_messages(
        self,
        session_id: str,
        user_message: str,
    ) -> list[ChatCompletionMessageParam]:
        recent_messages = self.memory.get_recent_messages(session_id=session_id, limit=8)
        related_notes = self.memory.search_notes(query=user_message, limit=5)

        notes_block = (
            "\n".join(f"- {item['note']}" for item in related_notes)
            if related_notes
            else "- No relevant notes found."
        )

        messages: list[ChatCompletionMessageParam] = []
        self._append_message(messages, "system", self._system_prompt())
        self._append_message(messages, "system", "Relevant saved notes:\n" + notes_block)

        for item in recent_messages:
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            self._append_message(messages, role, content)

        self._append_message(messages, "user", user_message)
        return messages

    def chat(self, user_message: str, session_id: str = "default") -> str:
        user_message = user_message.strip()
        if not user_message:
            return "Please type a message."

        messages = self._build_messages(
            session_id=session_id,
            user_message=user_message,
        )

        for _ in range(settings.max_tool_steps):
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                temperature=settings.temperature,
                messages=messages,
            )

            raw_content = response.choices[0].message.content
            if isinstance(raw_content, str):
                text = raw_content
            else:
                text = json.dumps(raw_content, ensure_ascii=False)

            try:
                action = self._extract_json(text)
            except Exception:
                self._append_message(messages, "assistant", text)
                self._append_message(
                    messages,
                    "user",
                    "That was not valid JSON. Return exactly one JSON object using the required schema.",
                )
                continue

            action_type = str(action.get("type", "")).strip()

            if action_type == "final":
                answer = str(action.get("answer", "")).strip() or "Done."
                remember_items = action.get("remember", [])

                if isinstance(remember_items, list):
                    for item in remember_items:
                        if isinstance(item, str) and item.strip():
                            self.memory.add_note(item.strip())

                self.memory.add_message(session_id, "user", user_message)
                self.memory.add_message(session_id, "assistant", answer)
                return answer

            if action_type == "tool":
                tool_name = str(action.get("tool", "")).strip()
                arguments = action.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}

                tool_result = self.tools.execute(tool_name, arguments)

                self._append_message(messages, "assistant", text)
                self._append_message(
                    messages,
                    "user",
                    (
                        "Tool result for "
                        + tool_name
                        + ":\n"
                        + json.dumps(tool_result, ensure_ascii=False, indent=2)
                        + "\nContinue. Return only one JSON object."
                    ),
                )
                continue

            self._append_message(messages, "assistant", text)
            self._append_message(
                messages,
                "user",
                "Invalid response schema. Return exactly one JSON object with type='tool' or type='final'.",
            )

        fallback = (
            "I reached the maximum number of tool steps. Try asking in a more focused way, "
            "or increase MAX_TOOL_STEPS in your .env file."
        )
        self.memory.add_message(session_id, "user", user_message)
        self.memory.add_message(session_id, "assistant", fallback)
        return fallback