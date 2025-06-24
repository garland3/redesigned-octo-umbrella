#!/usr/bin/env python3
"""
CLI Agent Runner - Command Line Interface for the AI Agent
Usage: python cli_agent.py [prompt]
If no prompt is provided, defaults to "tell me about this dir"
"""

import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import the unified agent system
from agent import AgentSession, AgentConfig, create_agent

# Load environment variables (don't override shell env vars)
load_dotenv(override=False)


class CLIAgentSession(AgentSession):
    """CLI version of AgentSession that outputs to console instead of WebSocket"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.log_file = Path("logs") / f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
    async def send_message(self, message_type: str, content: str, **kwargs):
        """Print messages to console instead of sending to WebSocket"""
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
            # Truncate long results for better readability
            if len(content) > 500:
                print(f"    {content[:500]}...")
            else:
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


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="CLI AI Agent")
    parser.add_argument("prompt", nargs="*", help="The prompt to send to the agent")
    parser.add_argument("--model", help="Override the model name")
    parser.add_argument("--max-loops", type=int, help="Maximum number of loops")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build prompt
    if args.prompt:
        user_prompt = " ".join(args.prompt)
    else:
        user_prompt = "tell me about this dir"
    
    print(f"🚀 Starting CLI Agent")
    print(f"📝 Prompt: {user_prompt}")
    
    # Create configuration overrides
    config_kwargs = {}
    if args.model:
        config_kwargs["model_name"] = args.model
    if args.max_loops:
        config_kwargs["max_loops"] = args.max_loops
    
    try:
        # Create agent and tool wrapper
        agent, tool_wrapper = await create_agent(
            agent_type="main",
            session_class=CLIAgentSession,
            **config_kwargs
        )
        
        # Initialize tool wrapper
        async with tool_wrapper:
            print(f"🔧 Loaded {len(tool_wrapper.get_openai_tools())} tools")
            
            # Run the agent
            result = await agent.run_agent_loop(user_prompt)
            
            print(f"\n✅ Agent completed successfully")
            print(f"📁 Session log: {agent.session.log_file}")
            
            return result
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return "Interrupted"
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


if __name__ == "__main__":
    asyncio.run(main())
