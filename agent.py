"""
Unified Agent System - Base classes for both web and CLI agents
This module provides a unified agent architecture that eliminates code duplication
between webapp.py and cli_agent.py
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
from typing import Dict, Any, List, Callable, Coroutine, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dotenv import load_dotenv

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

# Load environment variables (don't override shell env vars)
load_dotenv(override=False)


@dataclass
class AgentConfig:
    """Configuration for an agent instance"""
    model_name: str = "llama-3.3-70b-versatile"
    model_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    api_key: str = ""
    max_loops: int = 20
    max_context_characters: int = 15000
    temperature: float = 0.05
    tool_choice: str = "required"
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    agent_type: str = "main"
    
    def __post_init__(self):
        """Set default values based on environment variables"""
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY", "")
        
        if self.agent_type == "main":
            self.model_name = os.getenv("MAIN_MODEL_NAME", self.model_name)
            self.model_base_url = os.getenv("MAIN_MODEL_BASE_URL", self.model_base_url)
            self.api_key = os.getenv("MAIN_MODEL_API_KEY", self.api_key)
        elif self.agent_type == "sub":
            self.model_name = os.getenv("SUB_AGENT_MODEL_NAME", "llama-3.1-8b-instant")
            self.model_base_url = os.getenv("SUB_AGENT_MODEL_BASE_URL", self.model_base_url)
            self.api_key = os.getenv("SUB_AGENT_MODEL_API_KEY", self.api_key)
            self.max_loops = min(self.max_loops, 10)  # Limit sub-agent loops


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

    def get_openai_tools(self, tool_filter: List[str] = None) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas, optionally filtered by tool names"""
        if not self.all_tools_schema:
            raise RuntimeError("Tools not loaded. Use within an async context or after startup.")
        
        if tool_filter:
            # Filter tools based on the provided list
            filtered_tools = []
            for tool_schema in self.all_tools_schema:
                tool_name = tool_schema.get("function", {}).get("name", "")
                if tool_name in tool_filter:
                    filtered_tools.append(tool_schema)
            return filtered_tools
        
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


class AgentSession(ABC):
    """Abstract base class for agent sessions"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.log_file = Path("logs") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log_file.parent.mkdir(exist_ok=True)
        self.messages = []
        self.current_loop = 0
        self.original_user_request = ""
        self.agent_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    @abstractmethod
    async def send_message(self, message_type: str, content: str, **kwargs):
        """Send a message - implementation depends on interface (web/CLI)"""
        pass
        
    async def log_event(self, event_type: str, data: dict):
        """Log an event to the session log file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "event_type": event_type,
            "loop_count": self.current_loop,
            "data": data
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class Agent:
    """
    Unified Agent class that can be used for both main agents and sub-agents
    with different configurations and tool sets.
    """
    
    def __init__(self, config: AgentConfig, tool_wrapper: HybridToolWrapper, session: AgentSession):
        self.config = config
        self.tool_wrapper = tool_wrapper
        self.session = session
        self.active_sub_agents: Dict[str, 'Agent'] = {}
        
    async def make_api_call(self, payload: dict) -> dict:
        """Make API call to the configured language model"""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        agent_type = "Main Agent" if self.config.agent_type == "main" else "Sub-Agent"
        await self.session.send_message("thinking", f"🤖 {agent_type} ({self.config.model_name}) thinking...")
        
        async with aiohttp.ClientSession() as client_session:
            async with client_session.post(
                self.config.model_base_url, 
                headers=headers, 
                json=payload, 
                timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                return await response.json()

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool and return formatted result"""
        try:
            await self.session.log_event("tool_call", {"tool": tool_name, "args": arguments})
            result = await self.tool_wrapper.call_tool(tool_name, arguments)
            
            # Handle spawn_new_agent specially
            if tool_name == "spawn_new_agent" and isinstance(result, dict):
                if "error" in result:
                    return {"success": False, "result": f"❌ Error spawning agent: {result['error']}"}
                else:
                    # Create sub-agent and run it
                    spawn_result = await self.handle_spawn_new_agent(result)
                    return spawn_result
            
            if tool_name == "all_work_is_finished" and isinstance(result, dict) and result.get("finished"):
                return {"success": True, "result": result.get("response", "Work is complete."), "finished": True}
            
            await self.session.log_event("tool_result", {"tool": tool_name, "result": str(result)[:500]})
            return {"success": True, "result": f"✅ {tool_name}({arguments}) → {result}"}
        except Exception as e:
            stack = traceback.format_exc()
            await self.session.log_event("tool_error", {"tool": tool_name, "error": str(e) + stack})
            return {"success": False, "result": f"❌ Error in {tool_name}: {str(e)}"}

    async def handle_spawn_new_agent(self, agent_info: dict) -> dict:
        """Handle spawning a new sub-agent"""
        try:
            agent_id = agent_info["agent_id"]
            task = agent_info["task"]
            context = agent_info.get("context", "")
            agent_type = agent_info.get("agent_type", "search")
            config = agent_info.get("config", {})
            
            # Create sub-agent configuration
            sub_config = AgentConfig(
                agent_type="sub",
                max_loops=config.get("max_loops", 10),
                tools=config.get("tools", []),
                system_prompt=config.get("prompt", "You are a helpful sub-agent.")
            )
            
            # Create sub-agent session (same type as parent)
            sub_session = type(self.session)(sub_config)
            
            # Create sub-agent
            sub_agent = Agent(sub_config, self.tool_wrapper, sub_session)
            self.active_sub_agents[agent_id] = sub_agent
            
            await self.session.send_message("system", f"🔄 Spawning sub-agent [{agent_id}] for: {task}")
            
            # Run sub-agent
            result = await sub_agent.run_agent_loop(f"{task}\n\nContext: {context}")
            
            # Clean up
            if agent_id in self.active_sub_agents:
                del self.active_sub_agents[agent_id]
            
            await self.session.send_message("system", f"✅ Sub-agent [{agent_id}] completed")
            
            return {"success": True, "result": f"Sub-agent completed task: {result}"}
            
        except Exception as e:
            return {"success": False, "result": f"❌ Error running sub-agent: {str(e)}"}

    async def summarize_context(self, messages: list, max_summary_length: int = 500) -> list:
        """Summarize conversation context to reduce length."""
        if len(messages) <= 4:
            return messages
        
        try:
            await self.session.send_message("system", "🔄 Summarizing context to reduce length...")
            
            # Select all but the first (system) and last two messages for summarization
            context_to_summarize = messages[1:-2]
            context_content = []
            for msg in context_to_summarize:
                if isinstance(msg, dict) and 'content' in msg and msg['content']:
                    content = msg['content']
                    context_content.append(str(content) if not isinstance(content, str) else content)
            
            if not context_content:
                await self.session.log_event("summarization_skipped", {"reason": "no_valid_content"})
                return messages
            
            summary_prompt = f"""You are summarizing a conversation between an AI assistant and tools to help reduce context length.

            ORIGINAL USER REQUEST: {self.session.original_user_request}

            Your task: Summarize the following conversation context in {max_summary_length} characters or less. 
            Focus ONLY on information relevant to the original user request.
            Include: decisions made, tools used, key findings, current progress, and any important discoveries.
            Exclude: redundant information, failed attempts that didn't lead anywhere, and verbose explanations.

            Context to summarize: {json.dumps(context_content)[:2000]}

            Summary:"""
            
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": summary_prompt}],
                "temperature": 0.1
            }
            
            summary_response_data = await self.make_api_call(payload)
            
            summary = "Summary unavailable"
            if isinstance(summary_response_data, dict):
                choices = summary_response_data.get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        summary = message.get("content", "Summary unavailable")
            
            await self.session.log_event("context_summarized", {"summary": summary})
            await self.session.send_message("system", f"✅ Context summarized.")
            
            # Reconstruct messages with the summary
            return [
                messages[0],  # Original system prompt
                {"role": "assistant", "content": f"[CONTEXT SUMMARY]: {summary}"},
                *messages[-2:] # Last 2 messages
            ]
            
        except Exception as e:
            await self.session.log_event("summarization_error", {"error": str(e)})
            await self.session.send_message("error", f"❌ Summarization failed: {str(e)}")
            return messages

    async def run_agent_loop(self, user_input: str) -> str:
        """Main agent loop - returns final result"""
        await self.session.log_event("session_start", {
            "model": self.config.model_name,
            "agent_type": self.config.agent_type,
            "max_loops": self.config.max_loops
        })
        
        await self.session.send_message("system", f"🤖 AI Agent Started ({self.config.agent_type})")
        await self.session.send_message("system", f"Model: {self.config.model_name}")
        await self.session.send_message("system", f"Session log: {self.session.log_file}")
        
        self.session.original_user_request = user_input
        self.session.messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"Question/Task: {user_input}"}
        ]
        
        final_result = "No result obtained"
        
        for loop_count in range(self.config.max_loops):
            self.session.current_loop = loop_count + 1
            await self.session.send_message("loop_start", f"🔄 Loop {loop_count + 1}/{self.config.max_loops}")
            
            # Calculate total characters and summarize if context is too long
            total_chars = sum(len(json.dumps(msg)) for msg in self.session.messages)
            if total_chars > self.config.max_context_characters:
                self.session.messages = await self.summarize_context(self.session.messages)

            # Get available tools (filtered if specified in config)
            openai_tools = self.tool_wrapper.get_openai_tools(
                tool_filter=self.config.tools if self.config.tools else None
            )
            
            if not openai_tools:
                await self.session.send_message("error", "No tools available.")
                break

            payload = {
                "model": self.config.model_name,
                "messages": self.session.messages,
                "tools": openai_tools,
                "tool_choice": self.config.tool_choice,
                "temperature": self.config.temperature
            }
            
            try:
                resp_json = await self.make_api_call(payload)
                await self.session.log_event("api_response", resp_json)
            except Exception as e:
                await self.session.send_message("error", f"❌ API call failed: {str(e)}")
                continue
            
            message = resp_json.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls") or []
            
            if not tool_calls:
                content = message.get("content", "No tool calls were made. If the task is not complete, please try again.")
                await self.session.send_message("assistant_response", content)
                self.session.messages.append({"role": "user", "content": "No tool called. Please decide the next step or call 'all_work_is_finished'."})
                continue

            tool_call = tool_calls[0]
            try:
                func_info = tool_call['function']
                tool_name = func_info['name']
                args_str = func_info.get('arguments', '{}')
                arguments = json.loads(args_str) if args_str else {}
                
                await self.session.send_message("tool_call", f"Calling {tool_name}", tool=tool_name, args=arguments)
                
                tool_result = await self.call_tool(tool_name, arguments)
                success = tool_result.get("success", False)
                result_content = tool_result.get("result", "No result")
                
                await self.session.send_message("tool_result", result_content, success=success)
                
                # Add tool call and result to messages
                self.session.messages.append({"role": "assistant", "tool_calls": [tool_call]})
                self.session.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get('id', 'unknown'),
                    "name": tool_name,
                    "content": result_content
                })
                
                # Check if work is finished
                if tool_result.get("finished"):
                    final_result = result_content
                    await self.session.send_message("task_completed", "✅ Task completed successfully!")
                    break
                    
            except Exception as e:
                error_msg = f"❌ Error processing tool call: {str(e)}"
                await self.session.send_message("error", error_msg)
                self.session.messages.append({"role": "user", "content": error_msg})
        
        else:
            await self.session.send_message("warning", f"⚠️ Reached maximum loops ({self.config.max_loops})")
            final_result = "Task incomplete: Maximum loops reached"
        
        await self.session.send_message("final_output", final_result)
        return final_result


# Default tool set
DEFAULT_LOCAL_TOOLS = [
    intelligent_file_search, 
    list_files, 
    read_file, 
    all_work_is_finished, 
    spawn_new_agent
]

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI search and retrieval AI agent assistant that works in a loop. You can call tools when necessary. 
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


async def create_agent(
    agent_type: str = "main",
    session_class=None,
    local_tools: List[Callable] = None,
    mcp_targets: List[str] = None,
    system_prompt: str = None,
    **config_kwargs
) -> tuple[Agent, HybridToolWrapper]:
    """
    Factory function to create an agent with proper configuration
    
    Args:
        agent_type: "main" or "sub"
        session_class: Class to use for session (must inherit from AgentSession)
        local_tools: List of local tool functions
        mcp_targets: List of MCP target URLs
        system_prompt: Custom system prompt
        **config_kwargs: Additional configuration options
        
    Returns:
        Tuple of (Agent, HybridToolWrapper)
    """
    if local_tools is None:
        local_tools = DEFAULT_LOCAL_TOOLS
    
    if mcp_targets is None:
        mcp_targets = []
    
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        # Load overview if available
        try:
            with open("static/overview.rst") as f:
                overview_rst = f.read()
            system_prompt += f"\n\nHere is an overview of the software \n\n{overview_rst}"
        except FileNotFoundError:
            pass
    
    # Create configuration
    config = AgentConfig(
        agent_type=agent_type,
        system_prompt=system_prompt,
        **config_kwargs
    )
    
    # Create tool wrapper
    tool_wrapper = HybridToolWrapper(mcp_targets=mcp_targets, local_tools=local_tools)
    
    # Create session if class provided
    if session_class:
        session = session_class(config)
        agent = Agent(config, tool_wrapper, session)
        return agent, tool_wrapper
    else:
        return config, tool_wrapper
