from typing import Any, Mapping

from .base_tool import BaseTool, ToolCallResult, ToolRiskLevel
from pywen.tools.tool_manager import register_tool


@register_tool(name="finish", providers=["pywenswe"])
class FinishTool(BaseTool):
    name = "finish"
    display_name = "Finish"
    description = "Finish the current task and return a final message"
    parameter_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Final message to return",
            }
        },
        "required": ["message"],
    }
    risk_level = ToolRiskLevel.SAFE

    async def execute(self, **kwargs) -> ToolCallResult:
        message = kwargs.get("message", "")
        return ToolCallResult(
            call_id="",
            result=message,
        )

    def build(self, provider: str = "", func_type: str = "") -> Mapping[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameter_schema,
            },
        }
