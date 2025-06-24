#!/usr/bin/env python3
"""
CLI Agent Runner - Command Line Interface for the AI Agent
Usage: python cli_agent.py [prompt]
If no prompt is provided, defaults to "tell me about this dir"
"""

import sys
import traceback
import aiohttp
import os
import json
import subprocess
import asyncio
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable, Coroutine
import argparse

# --- Function Imports ---
from functions.helper import generate_schema
from functions.coerce_to_json import extract_function_json
from functions.file_system_fns import search_files_with_context, list_files, read_file
from functions.core_callable_fns import all_work_is_finished, spawn_new_agent
from functions.intellegent_file_search import intelligent_file_search

# --- HybridToolWrapper Class ---
from fastmcp import Client
from fastmcp.tools import Tool

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridToolWrapper:
    """
    A wrapper that unifies tool calling for remote FastMCP servers and
    local Python functions, exposing them all in an OpenAI-compatible schema.
    """
    def __init__(self, mcp_targets: List[str] = None, local_tools: List[Callable] = None):
        self.mcp_targets = mcp_targets or []
        self.local_tool_funcs = local_tools or []
        self.clients: Dict[str, Client] = {target: Client(target) for target in self.mcp_targets}
        self.local_tool_registry: Dict[str, Callable] = {func.__name__: func for func in self.local_tool_funcs}
        self.remote_tool_map: Dict[str, Client] = {}
        self.all_tools_schema: List[Dict] = []

    async def __aenter__(self):
        logging.info("Entering async context and preparing tools...")
        # FIX: Only try to connect to clients if there are any specified.
        if self.clients:
            logging.info(f"Connecting to {len(self.clients)} remote MCP client(s)...")
            await asyncio.gather(*(client.__aenter__() for client in self.clients.values()))
        
        await self._load_all_tools()
        logging.info("Tool wrapper is live and ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logging.info("Exiting async context and closing connections...")
        if self.clients:
            await asyncio.gather(*(client.__aexit__(exc_type, exc_val, exc_tb) for client in self.clients.values()))

    def _python_type_to_json_schema(self, py_type: Any) -> str:
        if py_type is str: return "string"
        if py_type in (int, float): return "number"
        if py_type is bool: return "boolean"
        if py_type is list: return "array"
        if py_type is dict: return "object"
        return "string"

    def _generate_schema_from_function(self, func: Callable) -> Dict:
        func_name = func.__name__
        description = inspect.getdoc(func) or "No description provided."
        sig = inspect.signature(func)
        parameters = {"type": "object", "properties": {}, "required": []}
        for param in sig.parameters.values():
            if param.name == 'self': continue
            param_type = self._python_type_to_json_schema(param.annotation)
            parameters["properties"][param.name] = {"type": param_type}
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param.name)
        return {"type": "function", "function": {"name": func_name, "description": description, "parameters": parameters}}

    async def _load_all_tools(self):
        logging.info("Loading tools from all sources...")
        all_schemas = []
        if self.clients:
            for target, client in self.clients.items():
                try:
                    remote_tools: List[Tool] = await client.list_tools()
                    for tool in remote_tools:
                        self.remote_tool_map[tool.name] = client
                        all_schemas.append({"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.parameters or {}}})
                    logging.info(f"Loaded {len(remote_tools)} tools from {target}")
                except Exception as e:
                    logging.error(f"Failed to load tools from {target}: {e}")

        logging.info(f"Generating schemas for {len(self.local_tool_funcs)} local tools...")
        for func in self.local_tool_funcs:
            schema = self._generate_schema_from_function(func)
            all_schemas.append(schema)
        logging.info("Finished generating local tool schemas.")
        self.all_tools_schema = all_schemas

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        if not self.all_tools_schema:
            raise RuntimeError("Tools not loaded. Use within an async context or after startup.")
        return self.all_tools_schema

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name in self.local_tool_registry:
            logging.info(f"Calling local tool: '{tool_name}' with args: {arguments}")
            func = self.local_tool_registry[tool_name]
            if asyncio.iscoroutinefunction(func):
                return await func(**arguments)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(**arguments))
        elif tool_name in self.remote_tool_map:
            client = self.remote_tool_map[tool_name]
            logging.info(f"Calling remote tool: '{tool_name}' on target '{client.target}'")
            result = await client.call_tool(tool_name, arguments)
            return result.text if hasattr(result, "text") and result.text is not None else result
        else:
            error_msg = f"Tool '{tool_name}' not found in local registry or any remote server."
            logging.error(error_msg)
            return {"error": error_msg}


class CLIAgentSession:
    """CLI version of AgentSession that outputs to console instead of WebSocket"""
    def __init__(self):
        self.log_file = Path("logs") / f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log_file.parent.mkdir(exist_ok=True)
        self.messages = []
        self.current_loop = 0
        self.original_user_request = ""
        
    async def log_event(self, event_type: str, data: dict):
        event = {"timestamp": datetime.now().isoformat(), "event": event_type, "data": data}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    async def send_message(self, message_type: str, content: str, **kwargs):
        """Print messages to console instead of sending to WebSocket"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if message_type == "system":
            print(f"[{timestamp}] 🔧 {content}")
        elif message_type == "loop_start":
            print(f"\n[{timestamp}] {content}")
        elif message_type == "tool_call":
            tool = kwargs.get('tool', '')
            args = kwargs.get('args', {})
            print(f"[{timestamp}] 🔧 Calling {tool}")
            if args:
                print(f"    Arguments: {json.dumps(args, indent=2)}")
        elif message_type == "tool_result":
            success = kwargs.get('success', True)
            status = "✅" if success else "❌"
            print(f"[{timestamp}] {status} Tool Result:")
            print(f"    {content}")
        elif message_type == "assistant_response":
            print(f"[{timestamp}] 🤖 Assistant: {content}")
        elif message_type == "task_completed":
            print(f"\n[{timestamp}] {content}")
        elif message_type == "final_output":
            print(f"[{timestamp}] 📋 Final Result:")
            print(f"    {content}")
        elif message_type == "error":
            print(f"[{timestamp}] ❌ Error: {content}")
        elif message_type == "warning":
            print(f"[{timestamp}] ⚠️  {content}")
        else:
            print(f"[{timestamp}] {content}")


# Configuration - environment variables
# Main Agent Configuration
MAIN_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "llama-3.3-70b-versatile")
MAIN_MODEL_BASE_URL = os.getenv("MAIN_MODEL_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# Sub-Agent Configuration
SUB_AGENT_MODEL_NAME = os.getenv("SUB_AGENT_MODEL_NAME", "llama-3.1-8b-instant")
SUB_AGENT_MODEL_BASE_URL = os.getenv("SUB_AGENT_MODEL_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# API Key Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAIN_MODEL_API_KEY = os.getenv("MAIN_MODEL_API_KEY", GROQ_API_KEY)
SUB_AGENT_MODEL_API_KEY = os.getenv("SUB_AGENT_MODEL_API_KEY", GROQ_API_KEY)

# Debug: Print API key status
print(f"Debug: GROQ_API_KEY loaded: {'Yes' if GROQ_API_KEY else 'No'}")
if not GROQ_API_KEY:
    print("⚠️ Warning: GROQ_API_KEY not found in environment!")

MAX_LOOP_COUNT = int(os.getenv("MAX_LOOP_COUNT", "15"))
MAX_CONTEXT_CHARACTERS = int(os.getenv("MAX_CONTEXT_CHARACTERS", "100000"))
TOOL_CHOICE = "required"

# System prompt
with open("static/overview.rst") as f:
    overview_rst = f.read()

SYSTEM_PROMPT = """You are a helpful AI search and retrieval AI agent assistant that works in a loop. You can call tools when necessary. 
After thinking, return in valid tool calling format. Call 'all_work_is_finished' when the task is complete.
Only call one tool per response/iteration. Be verbose and detailed in your final answers. Use the tools and answer this question or task about the engineering software. You can read the documention by using the tools. The documentation is in the sphinx format with .rst files. Keep working until you find the answer. To be perfectly clear you are trying to help the user by searching to find the answer in the given documentation.

SUB-AGENT CAPABILITIES:
You can spawn sub-agents to handle specific, focused tasks. This is useful for:
- Breaking down complex tasks into simpler pieces
- Searching specific files or directories
- Performing focused analysis on particular content

To spawn a sub-agent, use the 'spawn_new_agent' tool with:
- task_description: Clear, simple task (under 200 characters)
- agent_type: 'search', 'analysis', or 'simple_task'
- context: Any relevant context for the sub-agent

Sub-agents have limited tools and loops to keep them focused. Use them for well-defined, specific tasks."""

SYSTEM_PROMPT += f"\n\nHere is an overview of the software \n\n{overview_rst}"

LOCAL_TOOLS = [intelligent_file_search, list_files, read_file, all_work_is_finished, spawn_new_agent]
MCP_TARGETS = []

# Global state for CLI
CLI_STATE: Dict[str, Any] = {"tool_wrapper": None, "openai_tools": None}


async def make_api_call(session: CLIAgentSession, payload: dict, use_main_model: bool = True) -> dict:
    """Make API call to the language model"""
    headers = {
        "Authorization": f"Bearer {MAIN_MODEL_API_KEY if use_main_model else SUB_AGENT_MODEL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    base_url = MAIN_MODEL_BASE_URL if use_main_model else SUB_AGENT_MODEL_BASE_URL
    
    async with aiohttp.ClientSession() as client_session:
        async with client_session.post(base_url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API call failed with status {response.status}: {error_text}")
            return await response.json()


async def summarize_context(session: CLIAgentSession, messages: List[Dict]) -> List[Dict]:
    """Summarize context when it gets too long"""
    # Keep system message and recent messages, summarize the middle part
    if len(messages) <= 3:
        return messages
    
    system_msg = messages[0]
    recent_msgs = messages[-2:]
    middle_msgs = messages[1:-2]
    
    # Create summary prompt
    context_text = "\n".join([f"{msg['role']}: {msg.get('content', '')}" for msg in middle_msgs])
    summary_prompt = f"Please summarize this conversation context concisely:\n\n{context_text}"
    
    payload = {
        "model": SUB_AGENT_MODEL_NAME,
        "messages": [{"role": "user", "content": summary_prompt}],
        "temperature": 0.1
    }
    
    try:
        resp = await make_api_call(session, payload, use_main_model=False)
        summary = resp.get("choices", [{}])[0].get("message", {}).get("content", "Context summarized.")
        
        return [
            system_msg,
            {"role": "assistant", "content": f"[Previous context summary: {summary}]"},
            *recent_msgs
        ]
    except Exception as e:
        await session.log_event("summarization_error", {"error": str(e)})
        return messages


async def call_tool(session: CLIAgentSession, tool_name: str, arguments: dict) -> dict:
    """Call a tool and return formatted result"""
    try:
        wrapper = CLI_STATE.get("tool_wrapper")
        if not wrapper:
            return {"success": False, "result": "Tool wrapper not initialized"}
        
        result = await wrapper.call_tool(tool_name, arguments)
        
        if isinstance(result, dict) and "error" in result:
            return {"success": False, "result": result["error"]}
        
        # Convert result to string if it's not already
        if not isinstance(result, str):
            result = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
        
        return {"success": True, "result": result}
        
    except Exception as e:
        stack = traceback.format_exc()
        await session.log_event("tool_error", {"tool": tool_name, "error": str(e) + stack})
        return {"success": False, "result": f"❌ Error in {tool_name}: {str(e)}"}


async def run_agent_loop(session: CLIAgentSession, user_input: str):
    """Main agent loop - CLI version"""
    await session.log_event("session_start", {
        "main_model": MAIN_MODEL_NAME, 
        "sub_agent_model": SUB_AGENT_MODEL_NAME, 
        "max_loops": MAX_LOOP_COUNT
    })
    
    await session.send_message("system", f"🤖 CLI AI Agent Started")
    await session.send_message("system", f"Main Model: {MAIN_MODEL_NAME}")
    await session.send_message("system", f"Sub-Agent Model: {SUB_AGENT_MODEL_NAME}")
    await session.send_message("system", f"Session log: {session.log_file}")
    await session.send_message("system", f"User Request: {user_input}")
    
    session.original_user_request = user_input
    session.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question/Task: {user_input}"}
    ]
    
    for loop_count in range(MAX_LOOP_COUNT):
        session.current_loop = loop_count + 1
        await session.send_message("loop_start", f"🔄 Loop {loop_count + 1}/{MAX_LOOP_COUNT}")
        
        # Calculate total characters and summarize if context is too long
        total_chars = sum(len(json.dumps(msg)) for msg in session.messages)
        if total_chars > MAX_CONTEXT_CHARACTERS:
            await session.send_message("system", "Context too long, summarizing...")
            session.messages = await summarize_context(session, session.messages)

        openai_tools = CLI_STATE.get("openai_tools")
        if not openai_tools:
            await session.send_message("error", "Tool schemas not available.")
            return

        payload = {
            "model": MAIN_MODEL_NAME,
            "messages": session.messages,
            "tools": openai_tools,
            "tool_choice": TOOL_CHOICE,
            "temperature": 0.05
        }
        
        try:
            resp_json = await make_api_call(session, payload, use_main_model=True)
            await session.log_event("api_response", resp_json)
        except Exception as e:
            await session.send_message("error", f"❌ API call failed: {str(e)}")
            continue
        
        message = resp_json.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        
        if not tool_calls:
            content = message.get("content", "No tool calls were made. If the task is not complete, please try again.")
            await session.send_message("assistant_response", content)
            session.messages.append({"role": "user", "content": "No tool called. Please decide the next step or call 'all_work_is_finished'."})
            continue

        tool_call = tool_calls[0]
        try:
            func_info = tool_call['function']
            tool_name = func_info['name']
            arguments = json.loads(func_info.get('arguments', '{}'))
            
            await session.send_message("tool_call", f"🔧 Calling {tool_name}", tool=tool_name, args=arguments)
            tool_result = await call_tool(session, tool_name, arguments)
            
            if tool_result.get("finished"):
                await session.send_message("task_completed", "🎉 Task completed!")
                await session.send_message("final_output", tool_result["result"])
                return
            
            result_text = tool_result["result"]
            await session.send_message("tool_result", result_text, success=tool_result["success"])
            
            session.messages.append({"role": "assistant", "tool_calls": [tool_call]})
            session.messages.append({
                "role": "tool", 
                "tool_call_id": tool_call.get('id'), 
                "name": tool_name, 
                "content": result_text
            })
        except Exception as e:
            error_msg = f"❌ Tool processing error: {str(e)}"
            await session.send_message("error", error_msg)
            session.messages.append({"role": "user", "content": error_msg})
    
    await session.log_event("max_loops_reached", {})
    await session.send_message("warning", "⚠️ Maximum loops reached")


async def initialize_cli_agent():
    """Initialize the CLI agent with tools"""
    print("🔧 Initializing CLI Agent...")
    
    wrapper = HybridToolWrapper(mcp_targets=MCP_TARGETS, local_tools=LOCAL_TOOLS)
    await wrapper.__aenter__()
    CLI_STATE["tool_wrapper"] = wrapper
    CLI_STATE["openai_tools"] = wrapper.get_openai_tools()
    
    print(f"✅ CLI Agent initialized. Loaded {len(CLI_STATE['openai_tools'])} tools.")
    return wrapper


async def cleanup_cli_agent(wrapper: HybridToolWrapper):
    """Cleanup CLI agent resources"""
    print("\n🔧 Shutting down CLI Agent...")
    if wrapper:
        await wrapper.__aexit__(None, None, None)
    print("✅ CLI Agent shutdown complete.")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="AI Agent CLI")
    parser.add_argument("prompt", nargs='*', help="The prompt for the agent (default: 'tell me about this dir')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Join all prompt arguments or use default
    if args.prompt:
        user_prompt = " ".join(args.prompt)
    else:
        user_prompt = "tell me about this dir"
    
    print(f"🚀 Starting CLI Agent with prompt: '{user_prompt}'")
    print("=" * 60)
    
    wrapper = None
    try:
        # Initialize agent
        wrapper = await initialize_cli_agent()
        
        # Create session and run
        session = CLIAgentSession()
        await run_agent_loop(session, user_prompt)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logging.exception("CLI Agent error")
    finally:
        if wrapper:
            await cleanup_cli_agent(wrapper)


if __name__ == "__main__":
    asyncio.run(main())
