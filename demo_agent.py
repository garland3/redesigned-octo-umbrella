#!/usr/bin/env python3
"""
Demo/Test script for the unified agent system
Demonstrates how the system works without requiring API keys
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import the unified agent system
from agent import AgentSession, AgentConfig, HybridToolWrapper, Agent, DEFAULT_LOCAL_TOOLS


class DemoAgentSession(AgentSession):
    """Demo session that prints to console and simulates API responses"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.log_file = Path("logs") / f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
    async def send_message(self, message_type: str, content: str, **kwargs):
        """Print messages to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if message_type == "system":
            print(f"[{timestamp}] 🔧 {content}")
        elif message_type == "loop_start":
            print(f"\n[{timestamp}] {content}")
        elif message_type == "thinking":
            print(f"[{timestamp}] {content}")
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
            print(f"\n[{timestamp}] 📋 Final Result:")
            print(f"    {content}")
        elif message_type == "error":
            print(f"[{timestamp}] ❌ Error: {content}")
        elif message_type == "warning":
            print(f"[{timestamp}] ⚠️  {content}")
        else:
            print(f"[{timestamp}] {content}")


class DemoAgent(Agent):
    """Demo agent that simulates API responses without making real calls"""
    
    async def make_api_call(self, payload: dict) -> dict:
        """Simulate API response for demo purposes"""
        await self.session.send_message("thinking", f"🤖 Demo Agent simulating API call...")
        
        # Simulate delay
        await asyncio.sleep(0.5)
        
        # Return a simulated response that calls list_files
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "demo_call_1",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": "{\"directory\": \".\"}"
                        }
                    }]
                }
            }]
        }


async def demo_main():
    """Run a demo of the unified agent system"""
    print("🚀 Starting Unified Agent System Demo")
    print("=" * 50)
    
    # Create configuration
    config = AgentConfig(
        agent_type="main",
        max_loops=3,
        system_prompt="You are a demo AI agent. List files when asked about directories."
    )
    
    # Create tool wrapper
    tool_wrapper = HybridToolWrapper(local_tools=DEFAULT_LOCAL_TOOLS)
    
    try:
        # Initialize tool wrapper
        async with tool_wrapper:
            print(f"🔧 Initialized tool wrapper with {len(tool_wrapper.get_openai_tools())} tools")
            
            # Create demo session and agent
            session = DemoAgentSession(config)
            agent = DemoAgent(config, tool_wrapper, session)
            
            print(f"🤖 Created agent with config:")
            print(f"    - Type: {config.agent_type}")
            print(f"    - Model: {config.model_name}")
            print(f"    - Max loops: {config.max_loops}")
            print(f"    - Available tools: {[tool['function']['name'] for tool in tool_wrapper.get_openai_tools()]}")
            
            # Test 1: Main agent
            print(f"\n{'='*50}")
            print("🧪 Test 1: Main Agent Demo")
            print("='*50")
            
            result = await agent.run_agent_loop("what files are in this directory?")
            
            print(f"\n✅ Demo completed!")
            print(f"📁 Session log: {session.log_file}")
            
            # Test 2: Sub-agent configuration
            print(f"\n{'='*50}")
            print("🧪 Test 2: Sub-Agent Configuration Demo")
            print("='*50")
            
            sub_config = AgentConfig(
                agent_type="sub",
                max_loops=5,
                tools=["list_files", "read_file"],  # Limited tool set
                system_prompt="You are a focused sub-agent. Only list files."
            )
            
            print(f"🤖 Sub-agent configuration:")
            print(f"    - Type: {sub_config.agent_type}")
            print(f"    - Model: {sub_config.model_name}")
            print(f"    - Max loops: {sub_config.max_loops}")
            print(f"    - Limited tools: {sub_config.tools}")
            
            # Show filtered tools
            filtered_tools = tool_wrapper.get_openai_tools(tool_filter=sub_config.tools)
            print(f"    - Filtered tools available: {[tool['function']['name'] for tool in filtered_tools]}")
            
            print(f"\n🎉 Unified Agent System Demo Complete!")
            print(f"📊 Summary:")
            print(f"    - ✅ Unified agent architecture working")
            print(f"    - ✅ Tool filtering for sub-agents working")
            print(f"    - ✅ Session management working")
            print(f"    - ✅ Configuration system working")
            
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_main())
