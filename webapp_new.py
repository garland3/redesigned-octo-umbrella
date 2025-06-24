"""
Web Application - FastAPI WebSocket interface for the AI Agent
Uses the unified agent system from agent.py
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv

# Import the unified agent system
from agent import create_agent, DEFAULT_LOCAL_TOOLS
from web_session import WebAgentSession

# Load environment variables (don't override shell env vars)
load_dotenv(override=False)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FastAPI Application
app = FastAPI(title="AI Agent WebSocket Interface")

# Global state
APP_STATE: Dict[str, Any] = {"tool_wrapper": None}

# Configuration
MCP_TARGETS = []
MAX_LOOP_COUNT = int(os.getenv("MAX_LOOP_COUNT", 20))

# Active sessions tracking
ACTIVE_SESSIONS: Dict[str, WebAgentSession] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the tool wrapper on startup"""
    logging.info("Application starting up...")
    
    # Create and initialize tool wrapper
    _, tool_wrapper = await create_agent(
        agent_type="main",
        local_tools=DEFAULT_LOCAL_TOOLS,
        mcp_targets=MCP_TARGETS
    )
    
    # Initialize the tool wrapper
    await tool_wrapper.__aenter__()
    APP_STATE["tool_wrapper"] = tool_wrapper
    
    logging.info(f"Startup complete. Loaded {len(tool_wrapper.get_openai_tools())} tools.")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logging.info("Application shutting down...")
    tool_wrapper = APP_STATE.get("tool_wrapper")
    if tool_wrapper:
        await tool_wrapper.__aexit__(None, None, None)


@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the HTML interface"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Agent Interface</title>
        </head>
        <body>
            <h1>AI Agent WebSocket Interface</h1>
            <p>Connect via WebSocket to /ws</p>
            <p>Note: static/index.html not found - using fallback page</p>
        </body>
        </html>
        """)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for agent communication"""
    await websocket.accept()
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    session = None
    
    try:
        # Wait for initial message
        initial_data = await websocket.receive_json()
        user_input = initial_data.get("message", "")
        
        if not user_input:
            await websocket.send_json({
                "type": "error",
                "content": "No message provided",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        # Get tool wrapper
        tool_wrapper = APP_STATE.get("tool_wrapper")
        if not tool_wrapper:
            await websocket.send_json({
                "type": "error",
                "content": "Tool wrapper not initialized",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        # Create agent configuration for main agent
        from agent import AgentConfig
        config = AgentConfig(
            agent_type="main",
            max_loops=MAX_LOOP_COUNT
        )
        
        # Create session and agent
        session = WebAgentSession(config, websocket)
        ACTIVE_SESSIONS[session_id] = session
        
        from agent import Agent
        agent = Agent(config, tool_wrapper, session)
        
        # Send initial status
        await session.send_message("session_start", f"Session {session_id} started")
        
        # Run the agent
        result = await agent.run_agent_loop(user_input)
        
        # Send completion message
        await session.send_message("session_complete", f"Session completed: {result}")
        
    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logging.error(f"Error in WebSocket session {session_id}: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Session error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass  # Connection might be closed
    finally:
        # Clean up session
        if session_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[session_id]
        
        if session:
            logging.info(f"Session {session_id} completed. Log: {session.log_file}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    tool_wrapper = APP_STATE.get("tool_wrapper")
    return {
        "status": "healthy",
        "tools_loaded": len(tool_wrapper.get_openai_tools()) if tool_wrapper else 0,
        "active_sessions": len(ACTIVE_SESSIONS),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    tool_wrapper = APP_STATE.get("tool_wrapper")
    return {
        "active_sessions": len(ACTIVE_SESSIONS),
        "total_tools": len(tool_wrapper.get_openai_tools()) if tool_wrapper else 0,
        "log_directory": str(Path("logs").absolute()),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
