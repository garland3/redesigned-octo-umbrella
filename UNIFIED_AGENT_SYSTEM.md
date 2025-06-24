# Unified Agent System

This document explains the new unified agent architecture that eliminates code duplication between the webapp and CLI agent implementations.

## Overview

The previous implementation had significant code duplication between `webapp.py` and `cli_agent.py`, with both containing:
- Duplicate `HybridToolWrapper` classes
- Similar agent session management
- Duplicate API calling logic
- Similar loop management and tool calling

The new unified system provides a single, reusable agent architecture that both web and CLI interfaces can use.

## Architecture

### Core Components

1. **`agent.py`** - The main unified agent system containing:
   - `AgentConfig` - Configuration class for agent instances
   - `HybridToolWrapper` - Unified tool wrapper (no longer duplicated)
   - `AgentSession` - Abstract base class for agent sessions
   - `Agent` - Main agent class that handles the execution loop

2. **`web_session.py`** - WebSocket-specific session implementation
3. **`cli_agent_new.py`** - New CLI implementation using the unified system
4. **`webapp_new.py`** - New web application using the unified system

### Key Benefits

1. **No Code Duplication**: Common functionality is centralized in `agent.py`
2. **Flexible Configuration**: `AgentConfig` class allows easy customization
3. **Sub-Agent Support**: Sub-agents are just instances of `Agent` with different configurations
4. **Tool Filtering**: Sub-agents can have limited tool sets for focused tasks
5. **Interface Agnostic**: Same agent logic works with web, CLI, or any other interface

## Usage

### CLI Usage

```bash
# Basic usage
python cli_agent_new.py "what files are in this directory?"

# With options
python cli_agent_new.py "analyze the code structure" --model llama-3.3-70b-versatile --max-loops 10 --verbose
```

### Web Usage

```bash
# Start the web server
python webapp_new.py

# Connect via WebSocket to /ws
# Send JSON: {"message": "your query here"}
```

### Programmatic Usage

```python
from agent import create_agent, AgentConfig
from cli_agent_new import CLIAgentSession

# Create a main agent
agent, tool_wrapper = await create_agent(
    agent_type="main",
    session_class=CLIAgentSession,
    max_loops=20
)

# Initialize and run
async with tool_wrapper:
    result = await agent.run_agent_loop("your query here")
```

## Sub-Agent Architecture

Sub-agents are created using the same `Agent` class but with different configurations:

### Sub-Agent Types

1. **Search Agent** (`agent_type="search"`)
   - Tools: `intelligent_file_search`, `list_files`, `read_file`
   - Max loops: 8
   - Purpose: Focused file searching and content retrieval

2. **Analysis Agent** (`agent_type="analysis"`)
   - Tools: `read_file`, `list_files`
   - Max loops: 6
   - Purpose: Content analysis and interpretation

3. **Simple Task Agent** (`agent_type="simple_task"`)
   - Tools: `list_files`, `read_file`
   - Max loops: 5
   - Purpose: Basic file operations

### Creating Sub-Agents

Sub-agents are spawned using the `spawn_new_agent` tool:

```python
# Main agent can spawn a sub-agent
await agent.call_tool("spawn_new_agent", {
    "task_description": "Find all Python files in the functions directory",
    "agent_type": "search",
    "context": "Looking for function definitions"
})
```

## Configuration

### Environment Variables

```bash
# Main agent (powerful model for coordination)
MAIN_MODEL_NAME=llama-3.3-70b-versatile
MAIN_MODEL_BASE_URL=https://api.groq.com/openai/v1/chat/completions
MAIN_MODEL_API_KEY=your_api_key

# Sub-agent (efficient model for focused tasks)
SUB_AGENT_MODEL_NAME=llama-3.1-8b-instant
SUB_AGENT_MODEL_BASE_URL=https://api.groq.com/openai/v1/chat/completions
SUB_AGENT_MODEL_API_KEY=your_api_key

# Or use a single key for both
GROQ_API_KEY=your_groq_api_key
```

### AgentConfig Options

```python
config = AgentConfig(
    model_name="llama-3.3-70b-versatile",
    model_base_url="https://api.groq.com/openai/v1/chat/completions",
    api_key="your_key",
    max_loops=20,
    max_context_characters=15000,
    temperature=0.05,
    tool_choice="required",
    tools=["list_files", "read_file"],  # Filter available tools
    system_prompt="Custom system prompt",
    agent_type="main"  # or "sub"
)
```

## Migration Guide

### From Old CLI Agent

**Old:**
```bash
python cli_agent.py "query"
```

**New:**
```bash
python cli_agent_new.py "query"
```

### From Old WebApp

**Old:**
```bash
python webapp.py
```

**New:**
```bash
python webapp_new.py
```

## File Structure

```
├── agent.py              # Core unified agent system
├── cli_agent_new.py      # New CLI implementation
├── webapp_new.py         # New web implementation
├── web_session.py        # WebSocket session class
├── demo_agent.py         # Demo/test script
├── cli_agent.py          # Old CLI (can be removed)
├── webapp.py             # Old webapp (can be removed)
└── functions/            # Tool functions (unchanged)
    ├── core_callable_fns.py
    ├── file_system_fns.py
    ├── intelligent_file_search.py
    └── helper.py
```

## Testing

Run the demo to verify the system works:

```bash
python demo_agent.py
```

Test the CLI agent:

```bash
python cli_agent_new.py "test query" --max-loops 3
```

Test the web interface:

```bash
python webapp_new.py
# Then connect to http://localhost:8000
```

## Benefits of the Unified System

1. **Maintainability**: Changes to agent logic only need to be made in one place
2. **Consistency**: Both web and CLI interfaces behave identically
3. **Extensibility**: Easy to add new interfaces (Discord bot, Slack app, etc.)
4. **Testing**: Centralized logic is easier to test
5. **Sub-Agent Efficiency**: Sub-agents use less powerful models for cost optimization
6. **Tool Management**: Centralized tool registration and filtering

## Next Steps

1. Remove old files: `cli_agent.py` and `webapp.py`
2. Update any scripts that reference the old files
3. Consider adding more interface types (Discord, Slack, etc.)
4. Implement more sophisticated sub-agent types
5. Add monitoring and metrics collection
