AI Agent System with Sub-Agent Spawning
========================================

Overview
--------

This is an AI agent system designed to work effectively with less powerful LLMs by providing:

1. **Main Agent Loop**: Handles complex queries and coordination
2. **Sub-Agent Spawning**: Breaks down tasks into focused, simple sub-tasks
3. **Intelligent File Search**: Keyword-based search across directories
4. **Document Retrieval**: Access to documentation and file contents

Key Features
------------

Sub-Agent System
~~~~~~~~~~~~~~~~

The sub-agent system allows the main agent to spawn focused sub-agents for specific tasks:

- **Search Agents**: Find specific information in files/directories
- **Analysis Agents**: Analyze particular content with focus
- **Simple Task Agents**: Perform well-defined, single tasks

Each sub-agent has:
- Limited tool access for focus
- Reduced loop count (5-10 iterations)
- Simple, clear system prompts
- Specific task objectives under 200 characters

Tool Categories
~~~~~~~~~~~~~~~

**File Operations**:
- ``list_files``: List directory contents
- ``read_file``: Read file contents with line limits
- ``get_file_info``: Get file metadata

**Search Functions**:
- ``intelligent_file_search``: Keyword search across files
- ``search_in_specific_file``: Search within a single file
- ``find_files_by_pattern``: Find files matching patterns

**Agent Control**:
- ``spawn_new_agent``: Create focused sub-agents
- ``all_work_is_finished``: Complete tasks and return results

Best Practices for Less Powerful LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Keep Tasks Simple**: Break complex queries into smaller pieces
2. **Use Clear Language**: Avoid complex terminology in sub-agent tasks  
3. **Provide Context**: Give sub-agents specific, focused context
4. **Limit Scope**: Use the built-in loop and tool restrictions
5. **Validate Tasks**: Tasks should be under 200 characters and action-oriented

Example Usage Patterns
~~~~~~~~~~~~~~~~~~~~~~

**Search Pattern**::
    
    Main task: "Find authentication documentation"
    Sub-agent task: "Search for 'authentication' in docs directory"

**Analysis Pattern**::
    
    Main task: "Analyze API configuration options"
    Sub-agent task: "Read config.py and find API settings"

**Exploration Pattern**::
    
    Main task: "Find all Python files with error handling"
    Sub-agent task: "List files in src/ containing 'try' or 'except'"

System Architecture
------------------

The system uses a WebSocket interface with FastAPI backend, supporting:

- Real-time communication with the web interface
- Persistent logging of all agent activities
- Context summarization for long conversations
- Error handling and recovery mechanisms

The main agent coordinates sub-agents while maintaining overall task context,
making it suitable for less powerful LLMs that benefit from focused, smaller tasks.
