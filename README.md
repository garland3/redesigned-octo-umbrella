# AI Agent System with Sub-Agent Spawning

A FastAPI-based AI agent system designed to work effectively with less powerful LLMs by breaking down complex tasks into focused sub-agents. **Now supports different models for main coordination vs. sub-agent execution!**

## Key Features

### 🤖 Dual-Model Architecture
- **Main Agent**: Uses a more powerful model for complex reasoning and coordination
- **Sub-Agents**: Use optimized models for focused, simple tasks
- **Cost Optimization**: Expensive models only for coordination, cheaper models for execution
- **Flexible Configuration**: Mix and match any OpenAI-compatible APIs

### 🔄 Sub-Agent Spawning
- **Search Agents**: Focus on finding specific information
- **Analysis Agents**: Analyze particular content with clear objectives  
- **Simple Task Agents**: Perform well-defined, single tasks
- Limited tools and loops to maintain focus

### 🔍 Intelligent File Search
- Keyword-based search across directories
- Content matching with context lines
- File pattern matching with wildcards
- Designed for less powerful LLMs with simple interfaces

## Quick Start

1. **Setup with uv**:
   ```bash
   ./setup.sh
   ```

2. **Configure Models** (edit `.env`):
   ```bash
   # Recommended: Groq API for best performance
   cp .env.example .env
   # Edit with your Groq API key from https://console.groq.com/
   
   # Main Agent - Llama 3.1 70B for smart coordination
   MAIN_MODEL_NAME=llama-3.1-70b-versatile
   MAIN_MODEL_BASE_URL=https://api.groq.com/openai/v1/chat/completions
   MAIN_MODEL_API_KEY=gsk_your_groq_api_key_here
   
   # Sub-Agents - Llama 3.1 8B for fast execution
   SUB_AGENT_MODEL_NAME=llama-3.1-8b-instant
   SUB_AGENT_MODEL_BASE_URL=https://api.groq.com/openai/v1/chat/completions
   SUB_AGENT_MODEL_API_KEY=gsk_your_groq_api_key_here
   ```
   
   See `docs/groq-setup.md` for detailed configuration options.

3. **Run the Server**:
   ```bash
   ./run.sh
   ```

4. **Open Browser**: Navigate to `http://localhost:8007`

## Model Configuration Strategies

### 🏆 Hybrid Setup (Recommended)
```bash
# Smart coordination with local execution
MAIN_MODEL_NAME=gpt-4o-mini              # OpenAI for reasoning
SUB_AGENT_MODEL_NAME=llama3:8b           # Local Ollama for tasks
```
**Benefits**: Best reasoning + cost efficiency + speed

### 💰 Cost-Effective Cloud
```bash
# Balance cost and capability
MAIN_MODEL_NAME=gpt-4o-mini              # Good reasoning
SUB_AGENT_MODEL_NAME=gpt-3.5-turbo       # Reliable execution
```
**Benefits**: Predictable costs + reliable performance

### 🔒 Fully Local
```bash
# No API costs, complete privacy
MAIN_MODEL_NAME=llama3:70b               # Best local reasoning
SUB_AGENT_MODEL_NAME=llama3:8b           # Fast local execution
```
**Benefits**: No costs + privacy + offline capable

### 🌐 Cross-Platform
```bash
# Mix different providers
MAIN_MODEL_NAME=claude-3-haiku-20240307  # Anthropic coordination
SUB_AGENT_MODEL_NAME=gpt-3.5-turbo       # OpenAI execution
```
**Benefits**: Provider diversity + specialized strengths

## Using Sub-Agents

### When to Use Sub-Agents
- Breaking down complex search tasks
- Analyzing specific files with focused objectives
- Exploring directories for particular content
- Any task that can be clearly defined in under 200 characters

### Sub-Agent Types

#### Search Agent
```python
spawn_new_agent(
    task_description="Find 'authentication' examples in docs directory",
    agent_type="search",
    context="Looking for API authentication patterns"
)
```

#### Analysis Agent  
```python
spawn_new_agent(
    task_description="Analyze config.py for database settings",
    agent_type="analysis", 
    context="Need to understand database configuration"
)
```

#### Simple Task Agent
```python
spawn_new_agent(
    task_description="List all Python files containing 'class'",
    agent_type="simple_task",
    context="Finding class definitions"
)
```

## Best Practices for Less Powerful LLMs

### ✅ Good Sub-Agent Tasks
- "Search for X in directory Y" 
- "Read file Z and find configuration for W"
- "List files in directory A containing keyword B"

### ❌ Avoid Complex Tasks
- "Analyze the entire codebase architecture"
- "Compare and evaluate multiple implementation approaches" 
- "Synthesize information from many different sources"

### 🎯 Task Guidelines
- Keep task descriptions under 200 characters
- Use simple, action-oriented language (find, read, list, search)
- Provide specific context when possible
- Break complex tasks into smaller pieces

## Available Tools

### File Operations
- `list_files`: List directory contents with filtering
- `read_file`: Read file contents with line limits  
- `get_file_info`: Get file metadata and properties

### Search Functions
- `intelligent_file_search`: Keyword search across files and directories
- `search_in_specific_file`: Search within a single file with context
- `find_files_by_pattern`: Find files matching wildcard patterns

### Agent Control
- `spawn_new_agent`: Create focused sub-agents for specific tasks
- `all_work_is_finished`: Complete tasks and return final results

## Configuration

### Environment Variables
```bash
MODEL_NAME=llama3:8b                    # LLM model to use
MODEL_BASE_URL=http://localhost:11434/v1/chat/completions  # Model API endpoint
MAX_LOOP_COUNT=20                       # Maximum loops for main agent
MAX_CONTEXT_CHARACTERS=15000            # Context size before summarization
```

### Sub-Agent Limits
- **Search agents**: Maximum 8 loops
- **Analysis agents**: Maximum 6 loops  
- **Simple task agents**: Maximum 5 loops

## Architecture

```
Main Agent (webapp.py)
├── WebSocket Interface (real-time communication)
├── Tool Wrapper (unified tool access)
├── Sub-Agent Manager (spawning and coordination)
└── Context Summarization (for long conversations)

Sub-Agents (focused tasks)
├── Limited tool access
├── Reduced loop counts
├── Simple system prompts
└── Clear task objectives
```

## Example Usage

See `example_usage.py` for programmatic examples of:
- Basic tool usage
- Sub-agent spawning
- Task validation
- Best practice patterns

## Logging

All agent activities are logged to `logs/session_YYYYMMDD_HHMMSS.jsonl` with:
- Tool calls and results
- Sub-agent spawning and completion
- Error handling and recovery
- Performance metrics

## Contributing

This system is designed to be simple and extensible. To add new functionality:

1. Create focused, single-purpose tools
2. Keep interfaces simple for less powerful LLMs
3. Add comprehensive error handling
4. Include validation for task complexity
5. Test with actual less powerful models

The goal is to make AI agents accessible and effective even with limited computational resources.