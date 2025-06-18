import inspect

import inspect
from typing import get_origin, get_args

def python_type_to_json_schema_type(python_type) -> str:
    """Convert Python type annotations to JSON Schema type strings."""
    if python_type is None or python_type == type(None):
        return "null"
    elif python_type == int:
        return "integer"
    elif python_type == float:
        return "number"
    elif python_type == str:
        return "string"
    elif python_type == bool:
        return "boolean"
    elif python_type == list or get_origin(python_type) == list:
        return "array"
    elif python_type == dict or get_origin(python_type) == dict:
        return "object"
    else:
        # For complex types or unknown types, default to string
        return "string"

def generate_schema(function: callable) -> dict:
    """Generates a JSON Schema representing a Python function's structure.

    Args:
        function: The Python function to be analyzed for schema generation.

    Returns:
        A dictionary representing the JSON Schema of the function. 
    """

    schema = {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function) if inspect.getdoc(function) else "",
        }
    }

    # Extract parameters from the function signature
    signature = inspect.signature(function)
    properties = {}
    required_params = []
    
    for param_name, param in signature.parameters.items():
        param_type = param.annotation
        if param_type != inspect.Parameter.empty:
            # Convert Python type to JSON Schema type
            json_type = python_type_to_json_schema_type(param_type)
            properties[param_name] = {"type": json_type}
        else:
            # No type annotation available - use string as default valid type
            properties[param_name] = {"type": "string"}
        
        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required_params.append(param_name)
    
    schema["function"]["parameters"] = {
        "type": "object",  
        "properties": properties,
        "required": required_params
    }

    return schema



