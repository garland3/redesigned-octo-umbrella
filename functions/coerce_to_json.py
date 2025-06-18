"""
JSON extraction utilities for function calls.
"""
import json
import re
from typing import Dict, Any, Optional


def extract_function_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text that might contain function calls or other content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    
    if not text:
        return None
    
    # Try to find JSON in the text
    # Look for patterns like {"function": ...} or {"name": ...}
    
    # First try to parse the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Look for JSON-like patterns in the text
    json_patterns = [
        r'\{[^{}]*"function"[^{}]*\}',  # Function call pattern
        r'\{[^{}]*"name"[^{}]*\}',     # Name pattern
        r'\{[^{}]*"arguments"[^{}]*\}', # Arguments pattern
        r'\{.*\}',                      # Any JSON object
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try to extract from code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def clean_json_string(text: str) -> str:
    """
    Clean a string to make it more likely to be valid JSON.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    
    if not text:
        return text
    
    # Remove common issues
    text = text.strip()
    
    # Remove markdown code block markers
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text)
    
    # Fix common JSON issues
    text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
    text = re.sub(r',\s*]', ']', text)  # Remove trailing commas in arrays
    
    return text


def validate_function_call_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that a JSON object looks like a function call.
    
    Args:
        data: JSON data to validate
        
    Returns:
        Dict with validation results
    """
    
    if not isinstance(data, dict):
        return {"valid": False, "error": "Not a dictionary"}
    
    # Check for required function call fields
    required_patterns = [
        {"function": {"name": str, "arguments": dict}},  # OpenAI format
        {"name": str, "arguments": dict},                # Simple format
        {"tool": str, "args": dict},                     # Alternative format
    ]
    
    for pattern in required_patterns:
        if all(key in data for key in pattern.keys()):
            # Check types
            valid = True
            for key, expected_type in pattern.items():
                if isinstance(expected_type, dict):
                    # Nested validation
                    if not isinstance(data[key], dict):
                        valid = False
                        break
                    for nested_key, nested_type in expected_type.items():
                        if nested_key not in data[key] or not isinstance(data[key][nested_key], nested_type):
                            valid = False
                            break
                elif not isinstance(data[key], expected_type):
                    valid = False
                    break
            
            if valid:
                return {"valid": True, "format": "function_call"}
    
    return {"valid": False, "error": "Does not match function call pattern"}


def extract_tool_calls_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from an LLM response.
    
    Args:
        response_text: The response text from the LLM
        
    Returns:
        List of extracted tool calls
    """
    
    tool_calls = []
    
    if not response_text:
        return tool_calls
    
    # Look for JSON objects in the response
    json_objects = []
    
    # Try different extraction methods
    methods = [
        lambda x: [json.loads(x)],  # Entire text as JSON
        lambda x: [json.loads(match) for match in re.findall(r'\{.*?\}', x, re.DOTALL)],  # JSON objects
        lambda x: [json.loads(match) for match in re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', x, re.DOTALL)],  # Code blocks
    ]
    
    for method in methods:
        try:
            json_objects.extend(method(response_text))
        except:
            continue
    
    # Validate each JSON object as a potential tool call
    for obj in json_objects:
        validation = validate_function_call_json(obj)
        if validation["valid"]:
            tool_calls.append(obj)
    
    return tool_calls
