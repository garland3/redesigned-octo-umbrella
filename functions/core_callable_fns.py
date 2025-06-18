"""
Core callable functions for the AI agent system.
These functions provide basic agent control and sub-agent spawning capabilities.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


def all_work_is_finished(response: str) -> Dict[str, Any]:
    """
    Call this function when all work is completed and you have a final answer.
    
    Args:
        response: The final answer or summary of completed work
        
    Returns:
        Dict indicating work is finished with the response
    """
    return {
        "finished": True,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }


def spawn_new_agent(
    task_description: str, 
    context: str = "", 
    agent_type: str = "search",
    max_loops: int = 10
) -> Dict[str, Any]:
    """
    Spawn a new sub-agent to handle a specific task.
    Keep tasks simple and focused for less powerful LLMs.
    
    Args:
        task_description: Clear, specific task for the sub-agent (e.g., "Find information about X in file Y")
        context: Optional context to provide to the sub-agent
        agent_type: Type of agent - 'search', 'analysis', or 'simple_task'  
        max_loops: Maximum number of loops for the sub-agent (default: 10)
        
    Returns:
        Dict with sub-agent information and task details
    """
    
    # Validate task description is simple and clear
    if len(task_description) > 200:
        return {
            "error": "Task description too long. Keep it under 200 characters for sub-agents.",
            "suggestion": "Break down into smaller, simpler tasks."
        }
    
    # Create simple agent types with specific capabilities
    agent_configs = {
        "search": {
            "tools": ["intelligent_file_search", "list_files", "read_file"],
            "prompt": "You are a focused search agent. Find specific information and return it clearly.",
            "max_loops": min(max_loops, 8)  # Limit loops for search tasks
        },
        "analysis": {
            "tools": ["read_file", "list_files"],
            "prompt": "You are an analysis agent. Read and analyze specific content, then provide clear findings.",
            "max_loops": min(max_loops, 6)
        },
        "simple_task": {
            "tools": ["list_files", "read_file"],
            "prompt": "You are a simple task agent. Complete one specific task and report back.",
            "max_loops": min(max_loops, 5)
        }
    }
    
    if agent_type not in agent_configs:
        return {
            "error": f"Unknown agent type: {agent_type}",
            "available_types": list(agent_configs.keys())
        }
    
    config = agent_configs[agent_type]
    
    # Generate a simple agent ID
    import time
    agent_id = f"{agent_type}_{int(time.time())}"
    
    result = {
        "agent_id": agent_id,
        "task": task_description,
        "context": context,
        "agent_type": agent_type,
        "config": config,
        "status": "spawned",
        "timestamp": datetime.now().isoformat(),
        "instructions": f"""
TASK: {task_description}

CONTEXT: {context}

AGENT TYPE: {agent_type}
AVAILABLE TOOLS: {', '.join(config['tools'])}
MAX LOOPS: {config['max_loops']}

{config['prompt']}

Keep your responses simple and focused. When you find the answer or complete the task, 
call 'all_work_is_finished' with your findings.
"""
    }
    
    logging.info(f"Spawned sub-agent {agent_id} for task: {task_description}")
    
    return result


def create_simple_task(task_type: str, target: str, details: str = "") -> Dict[str, Any]:
    """
    Create a simple, well-defined task for sub-agents.
    
    Args:
        task_type: Type of task - 'find_in_file', 'list_directory', 'search_keyword'
        target: The target (file path, directory, keyword)
        details: Additional details about what to look for
        
    Returns:
        Dict with formatted task ready for sub-agent
    """
    
    task_templates = {
        "find_in_file": f"Find information about '{details}' in file {target}",
        "list_directory": f"List all files in directory {target} and look for {details}",
        "search_keyword": f"Search for keyword '{target}' and find {details}",
        "read_specific": f"Read file {target} and extract {details}"
    }
    
    if task_type not in task_templates:
        return {
            "error": f"Unknown task type: {task_type}",
            "available_types": list(task_templates.keys())
        }
    
    task_description = task_templates[task_type]
    
    return {
        "task_type": task_type,
        "task_description": task_description,
        "target": target,
        "details": details,
        "ready_for_spawn": True
    }


def validate_task_for_simple_llm(task_description: str) -> Dict[str, Any]:
    """
    Validate if a task is suitable for a less powerful LLM.
    
    Args:
        task_description: The task to validate
        
    Returns:
        Dict with validation results and suggestions
    """
    
    issues = []
    suggestions = []
    
    # Check length
    if len(task_description) > 200:
        issues.append("Task too long")
        suggestions.append("Break into smaller tasks under 200 characters")
    
    # Check complexity indicators
    complex_words = ["analyze", "compare", "evaluate", "synthesize", "complex", "intricate"]
    found_complex = [word for word in complex_words if word.lower() in task_description.lower()]
    
    if found_complex:
        issues.append(f"Complex language detected: {', '.join(found_complex)}")
        suggestions.append("Use simpler verbs like 'find', 'read', 'list', 'search'")
    
    # Check if task has clear goal
    action_words = ["find", "read", "list", "search", "get", "show"]
    has_action = any(word in task_description.lower() for word in action_words)
    
    if not has_action:
        issues.append("No clear action word found")
        suggestions.append("Start with clear action: find, read, list, search, get, or show")
    
    return {
        "is_suitable": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
        "score": max(0, 10 - len(issues) * 3)  # Simple scoring system
    }
