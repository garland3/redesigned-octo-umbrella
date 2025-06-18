"""
Example usage of the AI agent system with sub-agent spawning.
This shows how to use the system programmatically.
"""

import asyncio
import json
from webapp import HybridToolWrapper, LOCAL_TOOLS, MCP_TARGETS

async def example_agent_usage():
    """Example of how to use the agent system with sub-agents."""
    
    # Initialize the tool wrapper
    wrapper = HybridToolWrapper(mcp_targets=MCP_TARGETS, local_tools=LOCAL_TOOLS)
    
    async with wrapper:
        # Example 1: Simple file search
        print("=== Example 1: Simple File Search ===")
        search_result = await wrapper.call_tool("intelligent_file_search", {
            "search_term": "FastAPI",
            "directory": ".",
            "max_results": 5
        })
        print(f"Search result: {json.dumps(search_result, indent=2)}")
        
        # Example 2: Spawn a search sub-agent
        print("\n=== Example 2: Spawn Search Sub-Agent ===")
        sub_agent_config = await wrapper.call_tool("spawn_new_agent", {
            "task_description": "Find all Python files in the functions directory",
            "context": "Looking for function implementations",
            "agent_type": "search",
            "max_loops": 5
        })
        print(f"Sub-agent config: {json.dumps(sub_agent_config, indent=2)}")
        
        # Example 3: List files
        print("\n=== Example 3: List Files ===")
        file_list = await wrapper.call_tool("list_files", {
            "directory": "./functions",
            "max_files": 10
        })
        print(f"File list: {json.dumps(file_list, indent=2)}")
        
        # Example 4: Read a specific file
        print("\n=== Example 4: Read File ===")
        file_content = await wrapper.call_tool("read_file", {
            "file_path": "./webapp.py",
            "max_lines": 20
        })
        print(f"File content preview: {json.dumps(file_content, indent=2)}")


async def example_sub_agent_task_creation():
    """Example of creating well-formed sub-agent tasks."""
    
    print("=== Sub-Agent Task Creation Examples ===")
    
    # Import the functions we need
    from functions.core_callable_fns import (
        create_simple_task, 
        validate_task_for_simple_llm,
        spawn_new_agent
    )
    
    # Example tasks for different scenarios
    tasks = [
        {
            "name": "Good Search Task",
            "task": create_simple_task("search_keyword", "authentication", "in API documentation")
        },
        {
            "name": "Good File Analysis Task", 
            "task": create_simple_task("find_in_file", "config.py", "database settings")
        },
        {
            "name": "Good Directory Task",
            "task": create_simple_task("list_directory", "./docs", "tutorial files")
        }
    ]
    
    for example in tasks:
        print(f"\n{example['name']}:")
        print(f"  Task: {example['task']}")
        
        if example['task'].get('ready_for_spawn'):
            # Validate the task
            validation = validate_task_for_simple_llm(example['task']['task_description'])
            print(f"  Validation: {validation}")
            
            # Create the spawn configuration
            spawn_config = spawn_new_agent(
                task_description=example['task']['task_description'],
                context="Example task for demonstration",
                agent_type="search"
            )
            print(f"  Spawn config: {json.dumps(spawn_config, indent=4)}")


if __name__ == "__main__":
    print("Running AI Agent Examples...")
    
    # Run the basic tool examples
    asyncio.run(example_agent_usage())
    
    print("\n" + "="*50)
    
    # Run the sub-agent task creation examples
    asyncio.run(example_sub_agent_task_creation())
