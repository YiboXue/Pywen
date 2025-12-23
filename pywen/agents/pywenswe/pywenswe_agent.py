"""Pywen Agent implementation with streaming logic."""
import os,subprocess, json
import platform, shutil
from pathlib import Path
from typing import Dict, List, Any, AsyncGenerator,Mapping
from pywen.agents.base_agent import BaseAgent
from pywen.agents.agent_events import AgentEvent 
from pywen.agents.text_function_parser import TextFunctionParser, create_text_based_system_prompt
from pywen.llm.llm_basics import LLMMessage
from pywen.llm.llm_events import LLM_Events
from pywen.config.token_limits import TokenLimits
from pywen.utils.session_stats import session_stats
from pywen.llm.llm_basics import LLMResponse, ToolCall

SYSTEM_PROMPT = f"""You are PYWEN, an interactive CLI agent who is created by PAMPAS-Lab, specializing in software engineering tasks. Your primary goal is to help users safely and efficiently, adhering strictly to the following instructions and utilizing your available tools.

# Core Mandates
- **Safety First:** Always prioritize user safety and data integrity. Be cautious with destructive operations.
- **Tool Usage:** Use available tools when the user asks you to perform file operations, run commands, or interact with the system.
- **Precision:** Make targeted, minimal changes that solve the specific problem.
- **Explanation:** Provide clear explanations of what you're doing and why.

# Available Tools
"""

TOOL_PROMPT_SUFFIX = f"""
# Primary Workflows

## Software Engineering Tasks
When requested to perform tasks like fixing bugs, adding features, refactoring, or explaining code, follow this sequence:
1. **Understand:** Think about the user's request and the relevant context.
2. **Plan:** Build a coherent plan for how you intend to resolve the user's task.
3. **Implement:** Use the available tools to act on the plan.

## File Operations
- Use `str_replace_editor` to view, create, and edit files
- Use `bash` for system operations when needed

## Tone and Style (CLI Interaction)
- **Concise & Direct:** Adopt a professional, direct, and concise tone suitable for a CLI environment.
- **Minimal Output:** Aim for fewer than 3 lines of text output per response whenever practical.
- **Clarity over Brevity:** While conciseness is key, prioritize clarity for essential explanations.
- **No Chitchat:** Avoid conversational filler. Get straight to the action or answer.
- **Tools vs. Text:** Use tools for actions, text output only for communication.

## Security and Safety Rules
- **Explain Critical Commands:** Before executing commands that modify the file system or system state, provide a brief explanation.
- **Security First:** Always apply security best practices. Never introduce code that exposes secrets or sensitive information.

# Examples

Example 1:
User: Create a hello world Python script
Assistant: I'll create a hello world Python script for you.
[Uses str_replace_editor tool with command=create to create the script]

Example 2:
User: What's in the config file?
Assistant: [Uses str_replace_editor tool with command=view to read the config file and shows content]

Example 3:
User: Run the tests
Assistant: I'll run the tests for you.
[Uses bash tool to execute test command]

# Final Reminder
Your core function is efficient and safe assistance. Always prioritize user control and use tools when the user asks you to perform file operations or run commands. Use the finish tool with a concise final message when the task is complete.
"""

SANDBOX_MACOS_SEATBELT_PROMPT = """
# MacOS Seatbelt

You are running under macos seatbelt with limited access to files outside the project
directory or system temp directory, and with limited access to host system resources such as ports. If you encounter
failures that could be due to macos seatbelt (e.g. if a command fails with 'Operation not permitted' or similar error),
as you report the error to the user, also explain why you think it could be due to macos seatbelt, and how the user may
need to adjust their seatbelt profile."""

SANBOX_DEFAULT = """
# Sandbox
You are running in a sandbox container with limited access to files outside the project directory or system temp directory, 
and with limited access to host system resources such as ports. If you encounter failures that could be due to sandboxing 
(e.g. if a command fails with 'Operation not permitted' or similar error), when you report the error to the user, 
also explain why you think it could be due to sandboxing, and how the user may need to adjust their sandbox configuration.
"""

SANBOX_OUTSIDE = """
# Outside of Sandbox
You are running outside of a sandbox container, directly on the user's system. For critical commands that are particularly 
likely to modify the user's system outside of the project directory or system temp directory, as you explain the command to the user 
(per the Explain Critical Commands rule above), also remind the user to consider enabling sandboxing.
"""

GIT_INFO_BLOCK = """
    # Git Repository
    - The current working (project) directory is being managed by a git repository.
    - When asked to commit changes or prepare a commit, always start by gathering information using shell commands:
    - `git status` to ensure that all relevant files are tracked and staged, using `git add ...` as needed.
    - `git diff HEAD` to review all changes (including unstaged changes) to tracked files in work tree since last commit.
        - `git diff --staged` to review only staged changes when a partial commit makes sense or was requested by the user.
    - `git log -n 3` to review recent commit messages and match their style (verbosity, formatting, signature line, etc.)
    - Combine shell commands whenever possible to save time/steps, e.g. `git status && git diff HEAD && git log -n 3`.
    - Always propose a draft commit message. Never just ask the user to give you the full commit message.
    - Prefer commit messages that are clear, concise, and focused more on "why" and less on "what".
    - Keep the user informed and ask for clarification or confirmation where needed.
    - After each commit, confirm that it was successful by running `git status`.
    - If a commit fails, never attempt to work around the issues without being asked to do so.
    - Never push changes to a remote repository without being asked explicitly by the user.
    """

#SWE area without new app
BASE_PROMPT_DEFAULT = """
You are PYWEN, an interactive CLI agent who is created by PAMPAS-Lab, specializing in software engineering tasks. Your primary goal is to help users safely and efficiently, adhering strictly to the following instructions and utilizing your available tools.

# Core Mandates

- **Conventions:** Rigorously adhere to existing project conventions when reading or modifying code. Analyze surrounding code, tests, and configuration first.
- **Libraries/Frameworks:** NEVER assume a library/framework is available or appropriate. Verify its established usage within the project (check imports, configuration files like 'package.json', 'Cargo.toml', 'requirements.txt', 'build.gradle', etc., or observe neighboring files) before employing it.
- **Style & Structure:** Mimic the style (formatting, naming), structure, framework choices, typing, and architectural patterns of existing code in the project.
- **Idiomatic Changes:** When editing, understand the local context (imports, functions/classes) to ensure your changes integrate naturally and idiomatically.
- **Comments:** Add code comments sparingly. Focus on *why* something is done, especially for complex logic, rather than *what* is done. Only add high-value comments if necessary for clarity or if requested by the user. Do not edit comments that are separate from the code you are changing. *NEVER* talk to the user or describe your changes through comments.
- **Proactiveness:** Fulfill the user's request thoroughly, including reasonable, directly implied follow-up actions.
- **Confirm Ambiguity/Expansion:** Do not take significant actions beyond the clear scope of the request without confirming with the user. If asked *how* to do something, explain first, don't just do it.
- **Explaining Changes:** After completing a code modification or file operation *do not* provide summaries unless asked.
- **Path Construction:** Before using any file system tool (e.g., 'str_replace_editor'), you must construct the full absolute path for the file_path argument. Always combine the absolute path of the project's root directory with the file's path relative to the root. For example, if the project root is /path/to/project/ and the file is foo/bar/baz.txt, the final path you must use is /path/to/project/foo/bar/baz.txt. If the user provides a relative path, you must resolve it against the root directory to create an absolute path.
- **Do Not revert changes:** Do not revert changes to the codebase unless asked to do so by the user. Only revert changes made by you if they have resulted in an error or if the user has explicitly asked you to revert the changes.

# Primary Workflows

## Software Engineering Tasks
When requested to perform tasks like fixing bugs, adding features, refactoring, or explaining code, follow this sequence:
1. **Understand:** Think about the user's request and the relevant codebase context. Use 'bash' commands like `grep -r "pattern" .` and `find . -name "*.py"` extensively to understand file structures, existing code patterns, and conventions. Use 'str_replace_editor' with `command=view` to understand context and validate any assumptions you may have.
2. **Plan:** Build a coherent and grounded (based on the understanding in step 1) plan for how you intend to resolve the user's task. Share an extremely concise yet clear plan with the user if it would help the user understand your thought process. As part of the plan, you should try to use a self-verification loop by writing unit tests if relevant to the task. Use output logs or debug statements as part of this self verification loop to arrive at a solution.
3. **Implement:** Use the available tools to act on the plan, strictly adhering to the project's established conventions (detailed under 'Core Mandates'):
   - Use 'str_replace_editor' with `command=view` to view file content
   - Use 'str_replace_editor' with `command=str_replace` for precise edits
   - Use 'str_replace_editor' with `command=create` to create new files
   - Use 'bash' for searching (`grep -r`, `find`) and other system operations
4. **Verify (Tests):** If applicable and feasible, verify the changes using the project's testing procedures. Identify the correct test commands and frameworks by examining 'README' files, build/package configuration (e.g., 'package.json'), or existing test execution patterns. NEVER assume standard test commands.
5. **Verify (Standards):** VERY IMPORTANT: After making code changes, execute the project-specific build, linting and type-checking commands (e.g., 'tsc', 'npm run lint', 'ruff check .') that you have identified for this project (or obtained from the user). This ensures code quality and adherence to standards. If unsure about these commands, you can ask the user if they'd like you to run them and if so how to.

## Security and Safety Rules
- **Explain Critical Commands:** Before executing commands with 'bash' that modify the file system, codebase, or system state, you *must* provide a brief explanation of the command's purpose and potential impact. Prioritize user understanding and safety. You should not ask permission to use the tool; the user will be presented with a confirmation dialogue upon use (you do not need to tell them this).
- **Security First:** Always apply security best practices. Never introduce code that exposes, logs, or commits secrets, API keys, or other sensitive information.

## Tool Usage
- **File Paths:** Always use absolute paths when referring to files with tools like 'str_replace_editor'. Relative paths are not supported. You must provide an absolute path.
- **File Reading:** 
  - Use 'str_replace_editor' with `command=view` to view files
  - **IMPORTANT:** Use 'str_replace_editor' to view a file before editing it
- **File Editing:**
  - **Prefer 'str_replace_editor' for targeted changes:** Use `command=str_replace` with 'old_str' and 'new_str' for precise edits
  - **Use 'str_replace_editor' for new files:** Use `command=create` when creating a new file
  - **Preserve formatting:** When using 'str_replace_editor', ensure you preserve exact indentation (tabs/spaces) as it appears in the file
- **Completion:** Use the 'finish' tool with a concise final message once the task is complete.
  - **Make old_str unique:** The 'old_str' parameter must be unique in the file. Include enough surrounding context to ensure uniqueness
- **Command Execution:** Use the 'bash' tool for running shell commands, remembering the safety rule to explain modifying commands first.
- **Interactive Commands:** Try to avoid shell commands that are likely to require user interaction (e.g. `git rebase -i`). Use non-interactive versions of commands (e.g. `npm init -y` instead of `npm init`) when available, and otherwise remind the user that interactive shell commands are not supported and may cause hangs until canceled by the user.
- **Respect User Confirmations:** Most tool calls (also denoted as 'function calls') will first require confirmation from the user, where they will either approve or cancel the function call. If a user cancels a function call, respect their choice and do _not_ try to make the function call again. It is okay to request the tool call again _only_ if the user requests that same tool call on a subsequent prompt. When a user cancels a function call, assume best intentions from the user and consider inquiring if they prefer any alternative paths forward.

# MacOS Seatbelt
You are running under macos seatbelt with limited access to files outside the project directory or system temp directory, and with limited access to host system resources such as ports. If you encounter failures that could be due to macos seatbelt (e.g. if a command fails with 'Operation not permitted' or similar error), as you report the error to the user, also explain why you think it could be due to macos seatbelt, and how the user may need to adjust their seatbelt profile.

# Git Repository
- The current working (project) directory is being managed by a git repository.
- When asked to commit changes or prepare a commit, always start by gathering information using shell commands:
  - `git status` to ensure that all relevant files are tracked and staged, using `git add ...` as needed.
  - `git diff HEAD` to review all changes (including unstaged changes) to tracked files in work tree since last commit.
    - `git diff --staged` to review only staged changes when a partial commit makes sense or was requested by the user.
  - `git log -n 3` to review recent commit messages and match their style (verbosity, formatting, signature line, etc.)
- Combine shell commands whenever possible to save time/steps, e.g. `git status && git diff HEAD && git log -n 3`.
- Always propose a draft commit message. Never just ask the user to give you the full commit message.
- Prefer commit messages that are clear, concise, and focused more on "why" and less on "what".
- Keep the user informed and ask for clarification or confirmation where needed.
- After each commit, confirm that it was successful by running `git status`.
- If a commit fails, never attempt to work around the issues without being asked to do so.
- Never push changes to a remote repository without being asked explicitly by the user.

# Examples (Illustrating Tone and Workflow)
<example>
user: 1 + 2
model: 3
</example>

<example>
user: is 13 a prime number?
model: true
</example>

<example>
user: list files here.
model: ```
<function=bash>
<parameter=command>ls -la</parameter>
</function>
```

<example>
user: start the server implemented in server.js
model: Starting the server in background.
<function=bash>
<parameter=command>node server.js &</parameter>
</function>
```

<example>
user: Refactor the auth logic in src/auth.py to use the requests library instead of urllib.
model: I'll refactor 'src/auth.py'.
First, let me check for test coverage.
<function=bash>
<parameter=command>find . -name "*test*auth*" -o -name "auth*test*"</parameter>
</function>
<function=read_file>
<parameter=path>/path/to/tests/test_auth.py</parameter>
</function>
(After analysis)
Good, tests exist for auth. Let me verify 'requests' is available.
<function=bash>
<parameter=command>grep -E "^requests" requirements.txt || pip list | grep requests</parameter>
</function>
(After analysis)
'requests' is available.

Plan:
1. Replace 'urllib' calls with 'requests'
2. Add proper error handling
3. Remove old 'urllib' import
4. Run tests

Should I proceed?
user: Yes
model:
Let me read the current implementation.
<function=read_file>
<parameter=path>/path/to/src/auth.py</parameter>
</function>
(After reading the file)
Now I'll make targeted edits using the 'str_replace_editor' tool:
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/path/to/src/auth.py</parameter>
<parameter=old_str>import urllib.request
import urllib.error</parameter>
<parameter=new_str>import requests</parameter>
</function>
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/path/to/src/auth.py</parameter>
<parameter=old_str>    try:
        response = urllib.request.urlopen(url)
        return response.read()
    except urllib.error.URLError as e:
        print(f"Error: {e}")</parameter>
<parameter=new_str>    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error: {e}")</parameter>
</function>
Refactoring complete. Verifying...
<function=bash>
<parameter=command>ruff check src/auth.py && pytest tests/test_auth.py</parameter>
</function>
(After verification passes)
All checks passed.
Would you like me to commit these changes?
</example>

<example>
user: Write tests for someFile.ts
model:
I'll write tests for `someFile.ts`. First, let me read it.
<function=bash>
<parameter=command>find . -name "someFile.ts"</parameter>
</function>
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/path/to/someFile.ts</parameter>
</function>
Now checking existing test patterns.
<function=bash>
<parameter=command>find . -name "*.test.ts" -o -name "*.spec.ts" | head -5</parameter>
</function>
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/path/to/example.test.ts</parameter>
</function>
(After reviewing)
<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/path/to/someFile.test.ts</parameter>
<parameter=file_text>// Test code here
</parameter>
</function>
Tests written. Running them.
<function=bash>
<parameter=command>npm run test</parameter>
</function>
</example>

<example>
user: How do I update the user's profile information in this system?
model:
Let me search for profile update logic.
<function=bash>
<parameter=command>grep -r "updateProfile\|UserProfile\|editUser" --include="*.java" .</parameter>
</function>
(After reviewing search results)
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/path/to/UserProfileService.java</parameter>
</function>
(After reading)
The `updateUserProfile` method in `UserProfileService.java` handles this. It expects a user ID and a `UserProfileDTO` object...
</example>

<example>
user: Where are all the 'app.config' files in this project?
model:
<function=bash>
<parameter=command>find . -name "app.config"</parameter>
</function>
(After results)
Found:
- /path/to/moduleA/app.config
- /path/to/moduleB/app.config
Which one would you like to check, or should I read both?
</example>

# Final Reminder
# EFFICIENCY
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient bash commands like `find`, `grep -r`, and `git` commands with appropriate filters to minimize unnecessary operations.
Your core function is efficient and safe assistance. Balance extreme conciseness with the crucial need for clarity, especially regarding safety and potential system modifications. Always prioritize user control and project conventions. Never make assumptions about the contents of files; instead use 'str_replace_editor' with command=view to ensure you aren't making broad assumptions. Finally, use the finish tool when the user's query is completely resolved.
"""


RUNTIME_ENV_WINDOWS_PROMPT = """# Runtime Environment (IMPORTANT)
- OS: Windows ({release})
- Python: {python}
- Shell: {shell_hint} (COMSPEC={comspec})

# Command Rules (Windows)
- DO NOT output bash/zsh commands (no `ls`, `cat`, `grep`, `sed`, `awk`, `rm -rf`, or GNU-style pipes).
- Prefer PowerShell commands:
  - File listing: `Get-ChildItem`
  - Read file: `Get-Content`
  - Search text: `Select-String`
  - Delete files: `Remove-Item -Recurse -Force`
- Do NOT assume `/tmp`, `/proc`, `/dev`, or POSIX permissions.
- If Unix-like tools are required, explicitly ask whether Git Bash or WSL is available before using them.
"""

RUNTIME_ENV_MACOS_PROMPT = """# Runtime Environment (IMPORTANT)
- OS: macOS ({release})
- Python: {python}

# Command Rules (macOS)
- Use bash/zsh compatible commands.
- Do NOT assume GNU extensions (`sed -i`, `grep -P`, etc.) without verification.
- Prefer portable POSIX flags when possible.
"""

RUNTIME_ENV_LINUX_PROMPT = """# Runtime Environment (IMPORTANT)
- OS: Linux ({release})
- Python: {python}

# Command Rules (Linux)
- Use bash commands with typical GNU userland.
- Standard Linux filesystem layout is assumed unless otherwise stated.
"""

class PywenSWEAgent(BaseAgent):
    """Pywen SWE Agent with streaming iterative tool calling logic."""
    
    def __init__(self, config_mgr, cli, tool_mgr, use_text_format: bool = True):
        super().__init__(config_mgr, cli, tool_mgr)
        self.type = "PywenSWEAgent"
        self.use_text_format = use_text_format  # Use text-based function calling for models without native support
        session_stats.set_current_agent(self.type)
        self.current_turn_index = 0
        self.original_user_task = ""
        self.max_turns = self.config_mgr.get_app_config().max_turns
        self._should_stop = False
        self.system_prompt = self.get_core_system_prompt()
        self.conversation_history = self._update_system_prompt(self.system_prompt)
        self.file_metrics = {} 
    
    async def run(self, user_message: str) -> AsyncGenerator[AgentEvent, None]:
        """Run agent with streaming output and task continuation."""
        model_name = self.config_mgr.get_active_model_name() or ""
        max_tokens = TokenLimits.get_limit("qwen", model_name)
        self.cli.set_max_context_tokens(max_tokens)
        await self.setup_tools_mcp()
        self.original_user_task = user_message
        self.current_turn_index = 0
        self._should_stop = False
        session_stats.record_task_start(self.type)
        self.trajectory_recorder.start_recording(
            task=user_message,
            provider=self.config_mgr.get_active_agent().provider or "",
            model= model_name,
            max_steps=self.max_turns
        )
        yield AgentEvent.user_message(user_message, self.current_turn_index)

        self.conversation_history.append(LLMMessage(role="user", content=user_message))

        while self.current_turn_index < self.max_turns and not self._should_stop:
            async for event in self._process_turn_stream():
                yield event

    async def _process_turn_stream(self) -> AsyncGenerator[AgentEvent, None]:
        messages = [self._convert_single_message(msg) for msg in self.conversation_history]
        trajectory_msg = self.conversation_history.copy()
        tools = [tool.build("pywenswe") for tool in self.tool_mgr.list_for_provider("pywenswe")]
        completed_resp : LLMResponse = LLMResponse(content = "")

        tokens_used = sum(self.approx_token_count(m.content or "") for m in self.conversation_history)
        self.cli.set_current_tokens(tokens_used)
        
        # For text-based format, don't pass tools to LLM (they're in system prompt)
        llm_tools = None if self.use_text_format else tools
        
        async for event in self.llm_client.astream_response(messages= messages, tools= llm_tools, api = "chat"):
            if event.type == LLM_Events.REQUEST_STARTED:
                yield AgentEvent.llm_stream_start()
            elif event.type == LLM_Events.ASSISTANT_DELTA:
                yield AgentEvent.text_delta(str(event.data))
            elif event.type == LLM_Events.TOOL_CALL_DELTA:
                tc_data = event.data
                if tc_data is None:
                    continue
            elif event.type == LLM_Events.TOOL_CALL_READY:
                # 返回内容是tool_calls 字典列表
                # 1. 填充assistant LLMMessage
                tool_calls = event.data or {}
                tc_list = [ToolCall.from_raw(tc) for tc in tool_calls]
                assistant_msg = LLMMessage(
                    role="assistant",
                    tool_calls = tc_list,
                    content = "",
                )
                self.conversation_history.append(assistant_msg)
                # 2. 执行工具调用，拿到结果，填充tool LLMMessage
                async for tc_event in self._process_tool_calls(tc_list):
                    yield tc_event
            elif event.type == LLM_Events.TOKEN_USAGE:
                # 更新 token 使用统计
                usage = event.data or {}
                total = usage.get("total_tokens", 0)
                yield AgentEvent.turn_token_usage(total)
            elif event.type == LLM_Events.RESPONSE_FINISHED:
                self.current_turn_index += 1
                if not event.data:
                    continue
               # 处理结束状态
                finish_reason = event.data.get("finish_reason")
                completed_resp = LLMResponse.from_raw(event.data or {})
                
                # For text-based format, parse function calls from response content
                if self.use_text_format and completed_resp.content:
                    # Save original content for trajectory recording
                    original_content = completed_resp.content
                    remaining_text, parsed_tool_calls = TextFunctionParser.parse(completed_resp.content)
                    
                    if parsed_tool_calls:
                        # Update response content to remove function calls (for conversation history)
                        completed_resp.content = remaining_text
                        
                        # Add assistant message with parsed tool calls
                        # Use original_content for trajectory to preserve full model output
                        assistant_msg = LLMMessage(
                            role="assistant",
                            tool_calls=parsed_tool_calls,
                            content=original_content,  # Keep original content for trajectory
                        )
                        self.conversation_history.append(assistant_msg)
                        
                        # Execute the parsed tool calls
                        async for tc_event in self._process_tool_calls(parsed_tool_calls):
                            yield tc_event
                        
                        # Continue to next turn after tool execution
                        self.trajectory_recorder.record_llm_interaction(
                            messages=trajectory_msg,
                            response=completed_resp,
                            provider=self.config_mgr.get_active_agent().provider or "",
                            model=self.config_mgr.get_active_model_name() or "",
                            tools=tools,
                            agent_name=self.type,
                        )
                        continue
                    else:
                        # No tool calls found, treat as regular response
                        if remaining_text:
                            assistant_msg = LLMMessage(role="assistant", content=remaining_text)
                            self.conversation_history.append(assistant_msg)
                
                self.trajectory_recorder.record_llm_interaction(
                    messages= trajectory_msg,
                    response= completed_resp, 
                    provider=self.config_mgr.get_active_agent().provider or "",
                    model=self.config_mgr.get_active_model_name() or "",
                    tools=tools,
                    agent_name=self.type,
                )

                if finish_reason and finish_reason != "tool_calls":
                    yield AgentEvent.task_complete(finish_reason)

    def _convert_single_message(self, msg: LLMMessage) -> Dict[str, Any]:
        role = msg.role
        data: Dict[str, Any] = {"role": role}
        if role in ("system", "user"):
            data["content"] = msg.content or ""
            return data
    
        if role == "assistant":
            if msg.content is not None:
                data["content"] = msg.content
            if msg.tool_calls:
                converted_tool_calls = []
                for tc in msg.tool_calls:
                    converted_tool_calls.append({
                        "id": tc.call_id,
                        "type": tc.type or "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments or {}),
                        },
                    })
                data["tool_calls"] = converted_tool_calls
            return data

        if role == "tool":
            if not msg.tool_call_id:
                raise ValueError("Tool message must have tool_call_id")
            data["tool_call_id"] = msg.tool_call_id
            data["content"] = msg.content
            return data
    
        raise ValueError(f"Unsupported role for OpenAI messages: {role!r}")
 
    async def _process_tool_calls(self, tool_calls : List[ToolCall]) -> AsyncGenerator[AgentEvent, None]:
        # 2. 执行工具调用，拿到结果，填充tool LLMMessage
        for tc in tool_calls:
            tool = self.tool_mgr.get_tool(tc.name)
            if not tool:
                continue
            name = tc.name
            call_id = tc.call_id
            arguments = {}
            if isinstance(tc.arguments, Mapping):
                arguments = dict(tc.arguments)
            elif isinstance(tc.arguments, str) and tc.name == "apply_patch":
                arguments = {"input": tc.arguments}
           
            yield AgentEvent.tool_call(call_id, name, arguments)
            try:
                is_success, result = await self.tool_mgr.execute(name, arguments, tool)
                if not is_success:
                    msg = LLMMessage(role="tool", content= str(result), tool_call_id= call_id)
                    self.conversation_history.append(msg)
                    yield AgentEvent.tool_result(call_id, name, result, False, arguments)
                    continue
                yield AgentEvent.tool_result(call_id, name, result, True, arguments)
                content = result if isinstance(result, str) else json.dumps(result)
                tool_msg = LLMMessage(role="tool", content= content, tool_call_id=tc.call_id)
                self.conversation_history.append(tool_msg)
                if name == "finish":
                    self._should_stop = True
                    yield AgentEvent.task_complete("finish")
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                tool_msg = LLMMessage(role="tool", content= error_msg, tool_call_id=tc.call_id)
                self.conversation_history.append(tool_msg)
                yield AgentEvent.tool_result(call_id, name, error_msg, False, arguments)

    def _build_runtime_env_prompt(self) -> str:
        sys_name = platform.system()
        release = platform.release()
        python = platform.python_version()
    
        if sys_name == "Windows":
            comspec = os.environ.get("COMSPEC", "")
            ps = shutil.which("powershell") or shutil.which("pwsh")
            shell_hint = "PowerShell preferred" if ps else "cmd.exe"
    
            return RUNTIME_ENV_WINDOWS_PROMPT.format(
                release=release,
                python=python,
                shell_hint=shell_hint,
                comspec=comspec,
            )
    
        if sys_name == "Darwin":
            return RUNTIME_ENV_MACOS_PROMPT.format(
                release=release,
                python=python,
            )
    
        # Linux / other Unix
        return RUNTIME_ENV_LINUX_PROMPT.format(
            release=release,
            python=python,
        )

    def _update_system_prompt(self, system_prompt: str) -> List[LLMMessage]:
        cwd_prompt = (
            f"Please note that the user launched Pywen under the path {Path.cwd()}.\n"
            "All subsequent file-creation, file-writing, file-reading, and similar "
            "operations should be performed within this directory."
        )
        env_prompt = self._build_runtime_env_prompt()
        prompt = system_prompt.rstrip() + "\n\n" + env_prompt + "\n\n" + cwd_prompt

        # If using text-based format, add tool descriptions to system prompt
        if self.use_text_format:
            tools = [tool.build("pywenswe") for tool in self.tool_mgr.list_for_provider("pywenswe")]
            prompt = create_text_based_system_prompt(tools, prompt.rstrip())

        system_message = LLMMessage(role="system", content= prompt)
        return [system_message]

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        available_tools = self.tool_mgr.list_for_provider("pywenswe")
        system_prompt = SYSTEM_PROMPT 
        for tool in available_tools:
            system_prompt += f"- **{tool.name}**: {tool.description}\n"
            params = tool.parameter_schema.get('properties', {})
            if not params:
                continue
            param_list = ", ".join(params.keys())
            system_prompt += f"  Parameters: {param_list}\n"
        
        system_prompt += TOOL_PROMPT_SUFFIX 
        return system_prompt.strip()

    def get_core_system_prompt(self,user_memory: str = "") -> str:
        PYWEN_CONFIG_DIR = Path.home() / ".pywen"
        system_md_enabled = False
        system_md_path = (PYWEN_CONFIG_DIR / "system.md").resolve()
        system_md_var = os.environ.get("PYWEN_SYSTEM_MD", "").lower()
        if system_md_var and system_md_var not in ["0", "false"]:
            system_md_enabled = True
            if system_md_var not in ["1", "true"]:
                system_md_path = Path(system_md_var).resolve()
            if not system_md_path.exists():
                raise FileNotFoundError(f"Missing system prompt file '{system_md_path}'")

        def is_git_repository(path: Path) -> bool:
            """Check if the given path is inside a Git repository."""
            try:
                subprocess.run(
                    ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                return True
            except subprocess.CalledProcessError:
                return False

        def sandbox_info() -> str:
            if os.environ.get("SANDBOX") == "sandbox-exec":
                return SANDBOX_MACOS_SEATBELT_PROMPT 
            elif os.environ.get("SANDBOX"):
                return SANBOX_DEFAULT 
            else:
                return SANBOX_OUTSIDE 

        def git_info_block() -> str:
            if not is_git_repository(Path.cwd()):
                return "" 
            return  GIT_INFO_BLOCK

        base_prompt = system_md_path.read_text() if system_md_enabled else BASE_PROMPT_DEFAULT.strip()

        write_system_md_var = os.environ.get("PYWEN_WRITE_SYSTEM_MD", "").lower()
        if write_system_md_var and write_system_md_var not in ["0", "false"]:
            target_path = (
                system_md_path
                if write_system_md_var in ["1", "true"]
                else Path(write_system_md_var).resolve()
            )
            target_path.write_text(base_prompt)

        base_prompt += "\n" + sandbox_info()
        base_prompt += "\n" + git_info_block()

        if user_memory.strip():
            base_prompt += f"\n\n---\n\n{user_memory.strip()}"

        return base_prompt
