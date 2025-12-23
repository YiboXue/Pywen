"""
Text-based function call parser for models without native function calling support.

This module parses LLM text output in the format:
<function=TOOL_NAME>
<parameter=PARAM1>value1</parameter>
<parameter=PARAM2>value2</parameter>
</function>

and converts it into standard ToolCall objects that can be executed by the tool system.
"""

import re
import uuid
from typing import List, Dict, Any, Optional
from pywen.llm.llm_basics import ToolCall


class TextFunctionParser:
    """Parser for text-based function calls from LLM output."""
    
    # Regex pattern to match function blocks
    FUNCTION_PATTERN = re.compile(
        r'<function=([^>]+)>(.*?)</function>',
        re.DOTALL | re.IGNORECASE
    )
    
    # Regex pattern to match parameters within function blocks
    PARAMETER_PATTERN = re.compile(
        r'<parameter=([^>]+)>(.*?)</parameter>',
        re.DOTALL | re.IGNORECASE
    )
    
    @classmethod
    def parse(cls, text: str) -> tuple[str, List[ToolCall]]:
        """
        Parse text output to extract function calls and remaining text.
        
        Args:
            text: LLM output text containing function calls
            
        Returns:
            Tuple of (remaining_text, list of ToolCall objects)
            - remaining_text: Text content outside of function tags
            - tool_calls: List of parsed ToolCall objects
        """
        if not text:
            return "", []
        
        tool_calls: List[ToolCall] = []
        
        # Find all function blocks
        function_matches = cls.FUNCTION_PATTERN.finditer(text)
        
        # Track positions to extract remaining text
        last_end = 0
        text_parts = []
        
        for match in function_matches:
            # Extract text before this function call
            text_parts.append(text[last_end:match.start()])
            
            function_name = match.group(1).strip()
            function_content = match.group(2).strip()
            
            # Parse parameters within this function
            parameters = cls._parse_parameters(function_content)
            
            # Create ToolCall object with unique ID
            tool_call = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",  # Generate unique ID like: call_a1b2c3d4
                name=function_name,
                arguments=parameters,
                type="function"
            )
            tool_calls.append(tool_call)
            
            last_end = match.end()
        
        # Add any remaining text after last function
        text_parts.append(text[last_end:])
        
        # Join all text parts and clean up
        remaining_text = ''.join(text_parts).strip()
        
        return remaining_text, tool_calls
    
    @classmethod
    def _parse_parameters(cls, function_content: str) -> Dict[str, Any]:
        """
        Parse parameters from function content.
        
        Args:
            function_content: Content inside <function></function> tags
            
        Returns:
            Dictionary of parameter names to values
        """
        parameters = {}
        
        # Find all parameter blocks
        param_matches = cls.PARAMETER_PATTERN.finditer(function_content)
        
        for match in param_matches:
            param_name = match.group(1).strip()
            param_value = match.group(2).strip()
            
            # Try to convert value to appropriate type
            param_value = cls._convert_value(param_value)
            
            parameters[param_name] = param_value
        
        return parameters
    
    @classmethod
    def _convert_value(cls, value: str) -> Any:
        """
        Convert string value to appropriate Python type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value (str, int, float, bool, or original string)
        """
        # Handle boolean values
        if value.lower() in ('true'):
            return True
        if value.lower() in ('false'):
            return False
        
        # Try to convert to number
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Return as string if no conversion applies
        return value
    
    @classmethod
    def has_function_calls(cls, text: str) -> bool:
        """
        Check if text contains any function calls.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains function calls, False otherwise
        """
        return bool(cls.FUNCTION_PATTERN.search(text))
    
    @classmethod
    def format_tool_description(cls, tool_name: str, tool_schema: Dict[str, Any]) -> str:
        """
        Format tool description for inclusion in system prompt.
        
        Args:
            tool_name: Name of the tool
            tool_schema: Tool schema with description and parameters
            
        Returns:
            Formatted tool description with usage examples
        """
        description = tool_schema.get('description', '')
        parameters = tool_schema.get('parameters', {}).get('properties', {})
        required = tool_schema.get('parameters', {}).get('required', [])
        
        lines = [f"## {tool_name}"]
        lines.append(f"Description: {description}")
        lines.append("")
        lines.append("Usage:")
        lines.append(f"<function={tool_name}>")
        
        for param_name, param_info in parameters.items():
            param_desc = param_info.get('description', '')
            param_type = param_info.get('type', 'string')
            is_required = param_name in required
            required_mark = " (required)" if is_required else " (optional)"
            
            lines.append(f"<parameter={param_name}>{{{param_type}}}{required_mark}</parameter>  # {param_desc}")
        
        lines.append("</function>")
        lines.append("")
        
        return '\n'.join(lines)


def create_text_based_system_prompt(tools: List[Dict[str, Any]], base_prompt: str) -> str:
    """
    Create a system prompt that includes text-based tool calling instructions.
    
    Args:
        tools: List of tool schemas
        base_prompt: Base system prompt
        
    Returns:
        Complete system prompt with tool calling instructions
    """
    tool_descriptions = []
    
    for tool in tools:
        if 'function' in tool:
            tool_name = tool['function']['name']
            tool_schema = tool['function']
            description = TextFunctionParser.format_tool_description(tool_name, tool_schema)
            tool_descriptions.append(description)
        elif 'name' in tool:
            # Direct tool format
            description = TextFunctionParser.format_tool_description(tool['name'], tool)
            tool_descriptions.append(description)
    
    tool_section = '\n'.join(tool_descriptions)
    
    prompt = f"""{base_prompt}

# Tool Calling Format

When you need to use a tool, output it in the following XML-like format:

<function=TOOL_NAME>
<parameter=PARAM1>value1</parameter>
<parameter=PARAM2>value2</parameter>
</function>

**Important Rules:**
1. You MUST only call one tool in one response!
2. You can include explanatory text before or after function calls
3. Parameter values should be plain text (no quotes needed)
4. For boolean parameters, use: true/false
5. All required parameters must be included
6. Only include parameters defined in the tool schema
7. finish tool must be the last tool called

**Example:**

Let me check the current directory and list files.

<function=bash>
<parameter=command>pwd && ls -la</parameter>
</function>

# Available Tools

{tool_section}
"""
    
    return prompt
