"""
Web Agent Session - WebSocket-based session for web interface
"""

import json
import asyncio
from datetime import datetime
from fastapi import WebSocket
from agent import AgentSession, AgentConfig


class WebAgentSession(AgentSession):
    """WebSocket version of AgentSession for web interface"""
    
    def __init__(self, config: AgentConfig, websocket: WebSocket):
        super().__init__(config)
        self.websocket = websocket
        self.log_file = self.log_file.parent / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
    async def send_message(self, message_type: str, content: str, **kwargs):
        """Send message via WebSocket"""
        try:
            await self.websocket.send_json({
                "type": message_type, 
                "content": content, 
                "timestamp": datetime.now().isoformat(), 
                **kwargs
            })
            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the client
        except Exception as e:
            print(f"WebSocket send error: {e}")
    
    async def log_event(self, event_type: str, data: dict):
        """Override to also send log events to websocket"""
        # Call parent method to write to file
        await super().log_event(event_type, data)
        
        # Also send to websocket client
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "event_type": event_type,
            "loop_count": self.current_loop,
            "data": data
        }
        try:
            await self.websocket.send_json({"type": "log_event", "data": log_entry})
        except Exception as e:
            print(f"WebSocket log send error: {e}")
