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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv

# --- Function Imports ---
# Assuming these functions exist in the specified paths
from functions.helper import generate_schema
from functions.coerce_to_json import extract_function_json
from functions.file_system_fns import search_files_with_context, list_files, read_file
from functions.core_callable_fns import all_work_is_finished, spawn_new_agent
from functions.intellegent_file_search import intelligent_file_search

# --- HybridToolWrapper Class ---
# This class is now integrated directly into the application.
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

# --- FastAPI Application Setup ---

load_dotenv(override=True)
app = FastAPI(title="AI Agent WebSocket Interface")

APP_STATE: Dict[str, Any] = { "tool_wrapper": None, "openai_tools": None }

# Main Agent Configuration
MAIN_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "llama-3.1-70b-versatile")
MAIN_MODEL_BASE_URL = os.getenv("MAIN_MODEL_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# Sub-Agent Configuration (can use different, less powerful model)
SUB_AGENT_MODEL_NAME = os.getenv("SUB_AGENT_MODEL_NAME", "llama-3.1-8b-instant")
SUB_AGENT_MODEL_BASE_URL = os.getenv("SUB_AGENT_MODEL_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# API Key Configuration
# Use GROQ_API_KEY for both if using Groq, otherwise fall back to specific keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAIN_MODEL_API_KEY = os.getenv("MAIN_MODEL_API_KEY", GROQ_API_KEY)
SUB_AGENT_MODEL_API_KEY = os.getenv("SUB_AGENT_MODEL_API_KEY", GROQ_API_KEY)

# For backward compatibility with old OPENAI_API_KEY
if not MAIN_MODEL_API_KEY and not SUB_AGENT_MODEL_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MAIN_MODEL_API_KEY = OPENAI_API_KEY
    SUB_AGENT_MODEL_API_KEY = OPENAI_API_KEY

# General Configuration
MAX_LOOP_COUNT = int(os.getenv("MAX_LOOP_COUNT", 20))
MAX_CONTEXT_CHARACTERS = int(os.getenv("MAX_CONTEXT_CHARACTERS", 15000))
TOOL_CHOICE ="required"# "auto"

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

SYSTEM_PROMPT+=f"\n\nHere is an overview of the software \n\n{overview_rst}"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)



LOCAL_TOOLS = [intelligent_file_search, list_files, read_file, all_work_is_finished, spawn_new_agent]
MCP_TARGETS = []

# Sub-agent management
ACTIVE_SUB_AGENTS: Dict[str, Dict] = {}

@app.on_event("startup")
async def startup_event():
    logging.info("Application starting up...")
    
    wrapper = HybridToolWrapper(mcp_targets=MCP_TARGETS, local_tools=LOCAL_TOOLS)
    await wrapper.__aenter__()
    APP_STATE["tool_wrapper"] = wrapper
    APP_STATE["openai_tools"] = wrapper.get_openai_tools()
    logging.info(f"Startup complete. Loaded {len(APP_STATE['openai_tools'])} tools.")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutting down...")
    wrapper = APP_STATE.get("tool_wrapper")
    if wrapper:
        await wrapper.__aexit__(None, None, None)

class AgentSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.log_file = LOG_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.messages = []
        self.current_loop = 0
        self.original_user_request = ""
        
    async def log_event(self, event_type: str, data: dict):
        log_entry = { "timestamp": datetime.now().isoformat(), "event_type": event_type, "loop_count": self.current_loop, "data": data }
        with open(self.log_file, "a") as f: f.write(json.dumps(log_entry) + "\n")
        try:
            await self.websocket.send_json({"type": "log_event", "data": log_entry})
        except Exception as e:
            print(f"WebSocket send error: {e}")
    
    async def send_message(self, message_type: str, content: str, **kwargs):
        try:
            await self.websocket.send_json({"type": message_type, "content": content, "timestamp": datetime.now().isoformat(), **kwargs})
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"WebSocket send error: {e}")

async def summarize_context(session: AgentSession, messages: list, max_summary_length: int = 500):
    """Summarize conversation context to reduce length."""
    # if len(messages) <= 4:
    #     return messages
    
    try:
        print("Summaization invoked. ")
        await session.send_message("system", "🔄 Summarizing context to reduce length...")
        
        # Select all but the first (system) and last two messages for summarization
        context_to_summarize = messages[1:-2]
        context_content = []
        for msg in context_to_summarize:
            if isinstance(msg, dict) and 'content' in msg and msg['content']:
                content = msg['content']
                context_content.append(str(content) if not isinstance(content, str) else content)
        
        if not context_content:
            await session.log_event("summarization_skipped", {"reason": "no_valid_content"})
            return messages
        
        summary_prompt = f"""You are summarizing a conversation between an AI assistant and tools to help reduce context length.

        ORIGINAL USER REQUEST: {session.original_user_request}

        Your task: Summarize the following conversation context in {max_summary_length} characters or less. 
        Focus ONLY on information relevant to the original user request.
        Include: decisions made, tools used, key findings, current progress, and any important discoveries.
        Exclude: redundant information, failed attempts that didn't lead anywhere, and verbose explanations.

        Context to summarize: {json.dumps(context_content)[:2000]}

        Summary:"""
        
        payload = {
            "model": MAIN_MODEL_NAME,
            "messages": [{"role": "user", "content": summary_prompt}],
            "temperature": 0.1
        }
        
        summary_response_data = await make_api_call(session, payload, use_main_model=True)
        
        summary = "Summary unavailable"
        if isinstance(summary_response_data, dict):
            choices = summary_response_data.get("choices", [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    summary = message.get("content", "Summary unavailable")
        
        await session.log_event("context_summarized", {"summary": summary})
        await session.send_message("system", f"✅ Context summarized.")
        
        # Reconstruct messages with the summary
        return [
            messages[0],  # Original system prompt
            {"role": "assistant", "content": f"[CONTEXT SUMMARY]: {summary}"},
            *messages[-2:] # Last 2 messages
        ]
        
    except Exception as e:
        await session.log_event("summarization_error", {"error": str(e)})
        await session.send_message("error", f"❌ Summarization failed: {str(e)}")
        return messages


async def make_api_call(session: AgentSession, payload: dict, use_main_model: bool = True) -> dict:
    """Make API call to either main model or sub-agent model."""
    if use_main_model:
        model_url = MAIN_MODEL_BASE_URL
        api_key = MAIN_MODEL_API_KEY
        model_name = MAIN_MODEL_NAME
    else:
        model_url = SUB_AGENT_MODEL_BASE_URL
        api_key = SUB_AGENT_MODEL_API_KEY
        model_name = SUB_AGENT_MODEL_NAME
        # Ensure payload uses the correct model
        payload = payload.copy()
        payload["model"] = model_name
    
    headers = {"Content-Type": "application/json"}
    if api_key: 
        headers["Authorization"] = f"Bearer {api_key}"
    
    agent_type = "Main Agent" if use_main_model else "Sub-Agent"
    await session.send_message("thinking", f"🤖 {agent_type} ({model_name}) thinking...")
    
    async with aiohttp.ClientSession() as client_session:
        async with client_session.post(model_url, headers=headers, json=payload, timeout=60) as response:
            return await response.json()

async def call_tool(session: AgentSession, tool_name: str, arguments: dict) -> dict:
    wrapper = APP_STATE.get("tool_wrapper")
    if not wrapper:
        return {"success": False, "result": "Tool wrapper not initialized."}
    try:
        await session.log_event("tool_call", {"tool": tool_name, "args": arguments})
        result = await wrapper.call_tool(tool_name, arguments)
        
        # Handle spawn_new_agent specially
        if tool_name == "spawn_new_agent" and isinstance(result, dict):
            if "error" in result:
                return {"success": False, "result": f"❌ Error spawning agent: {result['error']}"}
            else:
                # Result is agent configuration, handle the spawning
                spawn_result = await handle_spawn_new_agent(session, {"success": True, "result": result})
                return spawn_result
        
        if tool_name == "all_work_is_finished" and isinstance(result, dict) and result.get("finished"):
            return {"success": True, "result": result.get("response", "Work is complete."), "finished": True}
        await session.log_event("tool_result", {"tool": tool_name, "result": str(result)[:500]})
        return {"success": True, "result": f"✅ {tool_name}({arguments}) → {result}"}
    except Exception as e:
        stack = traceback.format_exc()
        print(stack)
        await session.log_event("tool_error", {"tool": tool_name, "error": str(e) + stack})
        return {"success": False, "result": f"❌ Error in {tool_name}: {str(e)}"}

async def run_agent_loop(session: AgentSession, user_input: str):
    await session.log_event("session_start", {"main_model": MAIN_MODEL_NAME, "sub_agent_model": SUB_AGENT_MODEL_NAME, "max_loops": MAX_LOOP_COUNT})
    await session.send_message("system", f"🤖 AI Tool Assistant Started")
    await session.send_message("system", f"Main Model: {MAIN_MODEL_NAME}")
    await session.send_message("system", f"Sub-Agent Model: {SUB_AGENT_MODEL_NAME}")
    await session.send_message("system", f"Session log: {session.log_file}")
    
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
            session.messages = await summarize_context(session, session.messages)

        openai_tools = APP_STATE.get("openai_tools")
        if not openai_tools:
            await session.send_message("error", "Tool schemas not available.")
            return

        payload = { "model": MAIN_MODEL_NAME, "messages": session.messages, "tools": openai_tools, "tool_choice": TOOL_CHOICE, "temperature": 0.05 }
        
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
            session.messages.append({ "role": "tool", "tool_call_id": tool_call.get('id'), "name": tool_name, "content": result_text })
        except Exception as e:
            error_msg = f"❌ Tool processing error: {str(e)}"
            await session.send_message("error", error_msg)
            session.messages.append({"role": "user", "content": error_msg})
    
    await session.log_event("max_loops_reached", {})
    await session.send_message("warning", "⚠️ Maximum loops reached")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = AgentSession(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "start_task":
                asyncio.create_task(run_agent_loop(session, data["message"]))
            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await session.log_event("websocket_disconnect", {})
    except Exception as e:
        await session.log_event("websocket_error", {"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Serve the main interface from a static file."""
    # This assumes you have a 'static' directory with 'index.html' in it.
    static_file_path = Path("static/index.html")
    if static_file_path.is_file():
        with open(static_file_path, "r", encoding='utf-8') as f:
            return f.read()
    return HTMLResponse(content="<h1>Error: static/index.html not found.</h1><p>Please create the file to see the UI.</p>", status_code=404)

async def create_sub_agent_session(agent_config: Dict[str, Any], parent_session: AgentSession) -> 'SubAgentSession':
    """Create a new sub-agent session with limited capabilities."""
    return SubAgentSession(agent_config, parent_session)


class SubAgentSession:
    """A simplified agent session for sub-agents with limited tools and loops."""
    
    def __init__(self, agent_config: Dict[str, Any], parent_session: AgentSession):
        self.agent_id = agent_config["agent_id"]
        self.config = agent_config["config"]
        self.task = agent_config["task"]
        self.context = agent_config.get("context", "")
        self.parent_session = parent_session
        self.messages = []
        self.current_loop = 0
        self.max_loops = self.config["max_loops"]
        self.status = "active"
        self.result = None
        
        # Set up simple system prompt for sub-agent
        self.system_prompt = f"""You are a focused sub-agent. {self.config['prompt']}

TASK: {self.task}
CONTEXT: {self.context}

Available tools: {', '.join(self.config['tools'])}
Maximum loops: {self.max_loops}

Keep responses simple and focused. When you complete the task, call 'all_work_is_finished' with your findings."""

    async def log_event(self, event_type: str, data: dict):
        """Log sub-agent events to parent session."""
        await self.parent_session.log_event(f"sub_agent_{event_type}", {
            "agent_id": self.agent_id,
            "task": self.task,
            **data
        })
    
    async def send_message(self, message_type: str, content: str, **kwargs):
        """Send message through parent session with sub-agent prefix."""
        prefixed_content = f"[SUB-AGENT {self.agent_id}] {content}"
        await self.parent_session.send_message(message_type, prefixed_content, **kwargs)
    
    async def run(self) -> Dict[str, Any]:
        """Run the sub-agent with limited loops and tools."""
        await self.log_event("started", {"task": self.task, "model": SUB_AGENT_MODEL_NAME})
        await self.send_message("system", f"🤖 Sub-agent started (using {SUB_AGENT_MODEL_NAME}) for task: {self.task}")
        
        # Initialize messages
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Complete this task: {self.task}"}
        ]
        
        # Get available tools for this sub-agent
        wrapper = APP_STATE.get("tool_wrapper")
        if not wrapper:
            return {"success": False, "error": "Tool wrapper not available"}
        
        all_tools = wrapper.get_openai_tools()
        # Filter tools to only those allowed for this sub-agent
        allowed_tools = [tool for tool in all_tools 
                        if tool.get("function", {}).get("name") in self.config["tools"]]
        
        for loop_count in range(self.max_loops):
            self.current_loop = loop_count + 1
            await self.send_message("loop_start", f"🔄 Sub-agent loop {loop_count + 1}/{self.max_loops}")
            
            # Simple payload - no complex context management for sub-agents
            payload = {
                "model": SUB_AGENT_MODEL_NAME,
                "messages": self.messages[-10:],  # Keep only last 10 messages
                "tools": allowed_tools,
                "tool_choice": "required",
                "temperature": 0.1  # Lower temperature for more focused behavior
            }
            
            try:
                resp_json = await make_api_call(self.parent_session, payload, use_main_model=False)
                await self.log_event("api_response", {"response": str(resp_json)[:500]})
            except Exception as e:
                await self.send_message("error", f"❌ Sub-agent API call failed: {str(e)}")
                return {"success": False, "error": str(e)}
            
            message = resp_json.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls") or []
            
            if not tool_calls:
                content = message.get("content", "No tool called.")
                await self.send_message("assistant_response", content)
                self.messages.append({"role": "user", "content": "Please call a tool to complete the task or call 'all_work_is_finished'."})
                continue
            
            tool_call = tool_calls[0]
            try:
                func_info = tool_call['function']
                tool_name = func_info['name']
                arguments = json.loads(func_info.get('arguments', '{}'))
                
                # Check if tool is allowed for this sub-agent
                if tool_name not in self.config["tools"]:
                    await self.send_message("error", f"❌ Tool '{tool_name}' not allowed for this sub-agent")
                    continue
                
                await self.send_message("tool_call", f"🔧 Sub-agent calling {tool_name}", tool=tool_name, args=arguments)
                tool_result = await call_tool(self.parent_session, tool_name, arguments)
                
                if tool_result.get("finished"):
                    await self.send_message("task_completed", "🎉 Sub-agent task completed!")
                    self.status = "completed"
                    self.result = tool_result["result"]
                    await self.log_event("completed", {"result": self.result})
                    return {"success": True, "result": self.result, "agent_id": self.agent_id}
                
                result_text = tool_result["result"]
                await self.send_message("tool_result", result_text, success=tool_result["success"])
                
                self.messages.append({"role": "assistant", "tool_calls": [tool_call]})
                self.messages.append({"role": "tool", "tool_call_id": tool_call.get('id'), "name": tool_name, "content": result_text})
                
            except Exception as e:
                error_msg = f"❌ Sub-agent tool error: {str(e)}"
                await self.send_message("error", error_msg)
                self.messages.append({"role": "user", "content": error_msg})
        
        # Max loops reached
        await self.log_event("max_loops_reached", {})
        await self.send_message("warning", "⚠️ Sub-agent maximum loops reached")
        self.status = "timeout"
        return {"success": False, "error": "Maximum loops reached", "agent_id": self.agent_id}


# Helper functions for sub-agent creation
def create_search_agent_task(search_term: str, directory: str = ".") -> Dict[str, Any]:
    """Helper to create a simple search agent task."""
    return {
        "task_description": f"Search for '{search_term}' in directory {directory}",
        "agent_type": "search",
        "context": f"Looking for information about {search_term}"
    }

def create_file_analysis_task(file_path: str, focus: str) -> Dict[str, Any]:
    """Helper to create a file analysis task."""
    return {
        "task_description": f"Analyze file {file_path} focusing on {focus}",
        "agent_type": "analysis", 
        "context": f"Focus on finding information about: {focus}"
    }

def create_directory_exploration_task(directory: str, target: str) -> Dict[str, Any]:
    """Helper to create a directory exploration task."""
    return {
        "task_description": f"Explore directory {directory} to find {target}",
        "agent_type": "simple_task",
        "context": f"Looking for: {target}"
    }

async def handle_spawn_new_agent(session: AgentSession, tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """Handle spawning a new sub-agent and running it."""
    
    if not tool_result.get("success", False):
        return tool_result
    
    try:
        agent_config = tool_result["result"]
        
        # Validate agent config
        if not isinstance(agent_config, dict) or "agent_id" not in agent_config:
            return {"success": False, "result": "Invalid agent configuration"}
        
        agent_id = agent_config["agent_id"]
        ACTIVE_SUB_AGENTS[agent_id] = agent_config
        
        await session.send_message("system", f"🚀 Spawning sub-agent {agent_id}")
        await session.log_event("sub_agent_spawned", agent_config)
        
        # Create and run sub-agent
        sub_agent = await create_sub_agent_session(agent_config, session)
        sub_result = await sub_agent.run()
        
        # Clean up
        if agent_id in ACTIVE_SUB_AGENTS:
            del ACTIVE_SUB_AGENTS[agent_id]
        
        if sub_result["success"]:
            result_text = f"✅ Sub-agent {agent_id} completed successfully: {sub_result['result']}"
        else:
            result_text = f"❌ Sub-agent {agent_id} failed: {sub_result.get('error', 'Unknown error')}"
        
        await session.send_message("sub_agent_result", result_text)
        
        return {
            "success": True, 
            "result": result_text,
            "sub_agent_result": sub_result
        }
        
    except Exception as e:
        error_msg = f"Failed to handle sub-agent spawn: {str(e)}"
        await session.log_event("sub_agent_spawn_error", {"error": error_msg})
        return {"success": False, "result": error_msg}
