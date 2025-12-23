import os
from typing import Any, Mapping, Optional, Tuple

from .base_tool import BaseTool, ToolCallResult, ToolRiskLevel
from pywen.tools.tool_manager import register_tool


def _format_with_line_numbers(
    content: str,
    start_line: int,
    end_line: int,
) -> str:
    lines = content.splitlines()
    if not lines:
        return "(empty file)"
    total = len(lines)
    start = max(1, start_line)
    end = min(total, end_line)
    if start > end:
        start, end = 1, min(total, 200)
    pad = len(str(end))
    out_lines = []
    for i in range(start, end + 1):
        out_lines.append(f"{i:>{pad}}|{lines[i - 1]}")
    return "\n".join(out_lines)


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _preview_content(content: str, max_lines: int = 200) -> Tuple[str, int]:
    lines = content.splitlines()
    total = len(lines)
    end = min(total, max_lines)
    preview = _format_with_line_numbers(content, 1, end)
    return preview, total


def _format_with_catn_style(lines: list[str], start: int, end: int) -> str:
    """Format lines with cat -n style numbering (1-indexed)."""
    total = len(lines)
    start = max(1, start)
    end = min(total, end)
    if start > end:
        return ""
    pad = len(str(end))
    out = []
    for i in range(start, end + 1):
        out.append(f"{i:>{pad}}\t{lines[i - 1]}")
    return "\n".join(out)


def _find_snippet_range(content: str, needle: str, context: int = 2) -> Tuple[int, int]:
    """Locate a snippet range around the first occurrence of needle."""
    lines = content.splitlines()
    if not lines:
        return 1, 1
    needle_lines = needle.splitlines()
    if not needle_lines:
        return 1, min(len(lines), 1)
    first_line = needle_lines[0]
    idx = 0
    for i, line in enumerate(lines):
        if first_line in line:
            idx = i
            break
    start = max(1, idx + 1 - context)
    end = min(len(lines), idx + 1 + context + len(needle_lines) - 1)
    return start, end


STR_REPLACE_DESCRIPTION = """
View, create, and edit files using exact string replacement.

Commands:
- view: read a file and return a window of lines with line numbers
- create: create or overwrite a file with file_text
- str_replace: replace old_str with new_str in the file

Usage tips:
- Prefer view before editing to capture exact text and indentation.
- For str_replace, old_str must match exactly. Use replace_all=true to update all matches.
"""


@register_tool(name="str_replace_editor", providers=["pywenswe"])
class StrReplaceEditorTool(BaseTool):
    name = "str_replace_editor"
    display_name = "Str Replace Editor"
    description = "View, create, and edit files with exact string replacement"
    parameter_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "One of: view | create | str_replace",
            },
            "path": {
                "type": "string",
                "description": "Absolute path to the file",
            },
            "file_text": {
                "type": "string",
                "description": "File content for create",
            },
            "old_str": {
                "type": "string",
                "description": "Exact text to replace for str_replace",
            },
            "new_str": {
                "type": "string",
                "description": "Replacement text for str_replace",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default: false)",
                "default": False,
            },
            "start_line": {
                "type": "integer",
                "description": "Start line for view (default: 1)",
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "End line for view (default: 1000)",
                "default": 1000,
            },
            "overwrite": {
                "type": "boolean",
                "description": "Allow overwriting an existing file on create",
                "default": False,
            },
        },
        "required": ["command", "path"],
    }
    risk_level = ToolRiskLevel.MEDIUM

    async def _generate_confirmation_message(self, **kwargs) -> str:
        command = kwargs.get("command", "")
        path = kwargs.get("path", "")
        return f"📝 {self.display_name}: {command} -> {path}"

    async def execute(self, **kwargs) -> ToolCallResult:
        command = (kwargs.get("command") or "").strip().lower()
        path = kwargs.get("path")

        if not path:
            return ToolCallResult(call_id="", error="No path provided")

        if command == "view":
            if not os.path.exists(path):
                return ToolCallResult(call_id="", error=f"File not found: {path}")
            content = _read_file(path)
            start_line = int(kwargs.get("start_line", 1))
            end_line = int(kwargs.get("end_line", 1000))
            preview = _format_with_line_numbers(content, start_line, end_line)
            return ToolCallResult(
                call_id="",
                result=f"[File: {path}]\n{preview}",
            )

        if command == "create":
            file_text = kwargs.get("file_text")
            if file_text is None:
                return ToolCallResult(call_id="", error="Missing file_text for create")
            if os.path.exists(path) and not kwargs.get("overwrite", False):
                return ToolCallResult(
                    call_id="",
                    error=f"File already exists: {path} (use overwrite=true to replace)",
                )
            parent = os.path.dirname(path)
            if parent and not os.path.isdir(parent):
                return ToolCallResult(
                    call_id="",
                    error=f"Parent directory does not exist: {parent}",
                )
            _write_file(path, file_text)
            preview, total = _preview_content(file_text)
            return ToolCallResult(
                call_id="",
                result=f"File created: {path}\n[Preview: first {min(total, 200)} lines]\n{preview}",
            )

        if command == "str_replace":
            old_str = kwargs.get("old_str")
            new_str = kwargs.get("new_str")
            if old_str is None or new_str is None:
                return ToolCallResult(
                    call_id="",
                    error="Missing old_str or new_str for str_replace",
                )
            if not os.path.exists(path):
                return ToolCallResult(call_id="", error=f"File not found: {path}")
            content = _read_file(path)
            occurrences = content.count(old_str)
            if occurrences == 0:
                return ToolCallResult(
                    call_id="",
                    error="old_str not found in file",
                )
            replace_all = bool(kwargs.get("replace_all", False))
            if not replace_all and occurrences != 1:
                return ToolCallResult(
                    call_id="",
                    error=(
                        f"old_str matched {occurrences} times; "
                        "use replace_all=true or provide a more specific old_str"
                    ),
                )
            new_content = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
            _write_file(path, new_content)
            start, end = _find_snippet_range(new_content, new_str)
            lines = new_content.splitlines()
            snippet = _format_with_catn_style(lines, start, end)
            return ToolCallResult(
                call_id="",
                result=(
                    f"The file {path} has been edited. Here's the result of running "
                    f"`cat -n` on a snippet of {path}:\n"
                    f"{snippet}\n"
                    "Review the changes and make sure they are as expected. "
                    "Edit the file again if necessary."
                ),
            )

        return ToolCallResult(call_id="", error=f"Unknown command: {command}")

    def build(self, provider: str = "", func_type: str = "") -> Mapping[str, Any]:
        res = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description + "\n" + STR_REPLACE_DESCRIPTION.strip(),
                "parameters": self.parameter_schema,
            },
        }
        return res
