from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from huggingface_hub import HfApi

from config import settings
from memory import MemoryStore


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., dict[str, Any]]


def _safe_resolve_path(path_value: str | None) -> Path:
    relative_path = path_value or "."
    base = settings.workspace_dir.resolve()
    candidate = (base / relative_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError("Path must stay inside the workspace folder.")
    return candidate


def _safe_eval(expression: str) -> float:
    allowed_operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
        ast.FloorDiv: lambda a, b: a // b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }
    allowed_names = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in allowed_names:
            return float(allowed_names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return float(allowed_operators[type(node.op)](walk(node.left), walk(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return float(allowed_unary[type(node.op)](walk(node.operand)))
        raise ValueError("Unsupported expression.")

    tree = ast.parse(expression, mode="eval")
    return walk(tree)


class ToolRegistry:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.tools: dict[str, ToolDefinition] = {}
        self._register_defaults()

    def _register(self, tool: ToolDefinition) -> None:
        self.tools[tool.name] = tool

    def _register_defaults(self) -> None:
        self._register(
            ToolDefinition(
                name="list_files",
                description="List files inside the workspace. Use this before reading or writing when you need to inspect the folder.",
                parameters={
                    "path": "Relative path inside workspace. Default: '.'.",
                    "recursive": "Boolean. If true, include nested files up to a reasonable depth.",
                },
                handler=self.list_files,
            )
        )
        self._register(
            ToolDefinition(
                name="read_text_file",
                description="Read a UTF-8 text file from the workspace.",
                parameters={
                    "path": "Relative path inside workspace.",
                },
                handler=self.read_text_file,
            )
        )
        self._register(
            ToolDefinition(
                name="write_text_file",
                description="Write a UTF-8 text file inside the workspace.",
                parameters={
                    "path": "Relative path inside workspace.",
                    "content": "The file content to write.",
                    "overwrite": "Boolean. If false and the file exists, return an error.",
                },
                handler=self.write_text_file,
            )
        )
        self._register(
            ToolDefinition(
                name="calculator",
                description="Evaluate a simple math expression using +, -, *, /, **, %, //, pi, e, tau.",
                parameters={
                    "expression": "Example: '(24 * 7) / 3'.",
                },
                handler=self.calculator,
            )
        )
        self._register(
            ToolDefinition(
                name="save_note",
                description="Save a durable memory note for future chats.",
                parameters={
                    "note": "The note to store.",
                },
                handler=self.save_note,
            )
        )
        self._register(
            ToolDefinition(
                name="search_notes",
                description="Search durable memory notes that were saved earlier.",
                parameters={
                    "query": "Keywords to search for.",
                    "limit": "Maximum results to return.",
                },
                handler=self.search_notes,
            )
        )
        self._register(
            ToolDefinition(
                name="wikipedia_search",
                description="Search Wikipedia and return short summaries for public topics.",
                parameters={
                    "query": "Search phrase.",
                    "limit": "Maximum results to return.",
                },
                handler=self.wikipedia_search,
            )
        )
        self._register(
            ToolDefinition(
                name="huggingface_model_search",
                description="Search public Hugging Face models.",
                parameters={
                    "query": "What kind of model you want to search for.",
                    "limit": "Maximum results to return.",
                },
                handler=self.huggingface_model_search,
            )
        )
        self._register(
            ToolDefinition(
                name="run_python_code",
                description="Execute a short Python snippet inside the workspace sandbox. Disabled by default unless ALLOW_CODE_EXECUTION=true.",
                parameters={
                    "code": "Short Python code to execute.",
                },
                handler=self.run_python_code,
            )
        )

    def describe_tools_for_prompt(self) -> str:
        lines: list[str] = []
        for tool in self.tools.values():
            params = ", ".join(f"{key}: {value}" for key, value in tool.parameters.items())
            lines.append(f"- {tool.name}: {tool.description} Parameters -> {params}")
        return "\n".join(lines)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self.tools:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}

        try:
            result = self.tools[tool_name].handler(**arguments)
            return {"ok": True, "tool": tool_name, "result": result}
        except Exception as exc:
            return {"ok": False, "tool": tool_name, "error": str(exc)}

    def list_files(self, path: str = ".", recursive: bool = False) -> dict[str, Any]:
        target = _safe_resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        items: list[str] = []
        if target.is_file():
            items.append(str(target.relative_to(settings.workspace_dir)))
        else:
            iterator = target.rglob("*") if recursive else target.iterdir()
            for item in iterator:
                if ".sandbox" in item.parts:
                    continue
                if item.is_dir():
                    items.append(str(item.relative_to(settings.workspace_dir)) + "/")
                else:
                    items.append(str(item.relative_to(settings.workspace_dir)))

        items = sorted(items)[:300]
        return {"path": path, "items": items, "count": len(items)}

    def read_text_file(self, path: str) -> dict[str, Any]:
        target = _safe_resolve_path(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        content = target.read_text(encoding="utf-8")
        was_truncated = len(content) > settings.max_file_chars
        return {
            "path": path,
            "content": content[: settings.max_file_chars],
            "truncated": was_truncated,
        }

    def write_text_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        target = _safe_resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {path}. Call again with overwrite=true if you really want to replace it."
            )

        target.write_text(content, encoding="utf-8")
        return {"path": path, "bytes_written": len(content.encode("utf-8"))}

    def calculator(self, expression: str) -> dict[str, Any]:
        result = _safe_eval(expression)
        return {"expression": expression, "result": result}

    def save_note(self, note: str) -> dict[str, Any]:
        stored = self.memory.add_note(note)
        return {"saved": stored}

    def search_notes(self, query: str, limit: int = 5) -> dict[str, Any]:
        results = self.memory.search_notes(query=query, limit=limit)
        return {"query": query, "results": results}

    def wikipedia_search(self, query: str, limit: int = 3) -> dict[str, Any]:
        session = requests.Session()
        opensearch = session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": min(max(limit, 1), 5),
                "namespace": 0,
                "format": "json",
            },
            timeout=15,
        )
        opensearch.raise_for_status()
        _, titles, _, urls = opensearch.json()

        results: list[dict[str, Any]] = []
        for title, url in list(zip(titles, urls))[:limit]:
            summary_response = session.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                timeout=15,
            )
            if summary_response.ok:
                payload = summary_response.json()
                results.append(
                    {
                        "title": payload.get("title", title),
                        "summary": payload.get("extract", ""),
                        "url": payload.get("content_urls", {}).get("desktop", {}).get("page", url),
                    }
                )
            else:
                results.append({"title": title, "summary": "", "url": url})

        return {"query": query, "results": results}

    def huggingface_model_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        api = HfApi()
        models = api.list_models(search=query, limit=min(max(limit, 1), 10), sort="downloads", direction=-1)
        results: list[dict[str, Any]] = []
        for model in models:
            results.append(
                {
                    "id": model.id,
                    "downloads": getattr(model, "downloads", None),
                    "likes": getattr(model, "likes", None),
                    "pipeline_tag": getattr(model, "pipeline_tag", None),
                    "library_name": getattr(model, "library_name", None),
                }
            )
        return {"query": query, "results": results}

    def run_python_code(self, code: str) -> dict[str, Any]:
        if not settings.allow_code_execution:
            raise PermissionError(
                "Code execution is disabled. Set ALLOW_CODE_EXECUTION=true in .env to enable this tool."
            )

        sandbox_dir = settings.workspace_dir / ".sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=sandbox_dir,
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(code)
            temp_file = Path(handle.name)

        completed = subprocess.run(
            [sys.executable, str(temp_file)],
            cwd=settings.workspace_dir,
            capture_output=True,
            text=True,
            timeout=12,
        )

        return {
            "return_code": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
            "script_path": str(temp_file.relative_to(settings.workspace_dir)),
        }
