"""
File system functions for basic file operations.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def list_files(directory: str = ".", include_hidden: bool = False, max_files: int = 50) -> Dict[str, Any]:
    """
    List files and directories in a given path.
    
    Args:
        directory: Directory to list (default: current directory)
        include_hidden: Whether to include hidden files/directories
        max_files: Maximum number of files to return
        
    Returns:
        Dict with file listing information
    """
    
    try:
        if not os.path.exists(directory):
            return {"error": f"Directory '{directory}' does not exist"}
        
        if not os.path.isdir(directory):
            return {"error": f"'{directory}' is not a directory"}
        
        items = []
        count = 0
        
        for item in os.listdir(directory):
            if count >= max_files:
                break
                
            # Skip hidden files unless requested
            if not include_hidden and item.startswith('.'):
                continue
            
            item_path = os.path.join(directory, item)
            is_dir = os.path.isdir(item_path)
            
            try:
                stat_info = os.stat(item_path)
                size = stat_info.st_size if not is_dir else None
                
                items.append({
                    "name": item,
                    "path": item_path,
                    "type": "directory" if is_dir else "file",
                    "size": size,
                    "extension": Path(item).suffix if not is_dir else None
                })
                count += 1
                
            except OSError:
                # Skip items we can't access
                continue
        
        return {
            "directory": directory,
            "items": items,
            "total_items": len(items),
            "truncated": count >= max_files
        }
        
    except Exception as e:
        return {"error": f"Failed to list directory: {str(e)}"}


def read_file(file_path: str, max_lines: int = 100, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (for large files)
        encoding: File encoding (default: utf-8)
        
    Returns:
        Dict with file contents and metadata
    """
    
    try:
        if not os.path.exists(file_path):
            return {"error": f"File '{file_path}' does not exist"}
        
        if not os.path.isfile(file_path):
            return {"error": f"'{file_path}' is not a file"}
        
        # Get file info
        stat_info = os.stat(file_path)
        file_size = stat_info.st_size
        
        # Check if file is too large (> 1MB)
        if file_size > 1024 * 1024:
            return {
                "error": f"File too large ({file_size} bytes). Use a smaller max_lines value.",
                "file_size": file_size,
                "suggestion": "Try reading with max_lines=20 for large files"
            }
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            if max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip('\n\r'))
                content = '\n'.join(lines)
                truncated = i >= max_lines - 1
            else:
                content = f.read()
                lines = content.split('\n')
                truncated = False
        
        return {
            "file": file_path,
            "content": content,
            "lines_read": len(lines) if max_lines else len(content.split('\n')),
            "file_size": file_size,
            "truncated": truncated,
            "encoding": encoding
        }
        
    except UnicodeDecodeError:
        return {"error": f"Cannot decode file '{file_path}' with encoding '{encoding}'. File may be binary."}
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}


def search_files_with_context(
    search_term: str, 
    directory: str = ".",
    file_pattern: str = "*.py",
    context_lines: int = 2
) -> Dict[str, Any]:
    """
    Search for a term in files with context lines.
    
    Args:
        search_term: Term to search for
        directory: Directory to search in
        file_pattern: File pattern to match (e.g., "*.py", "*.txt")
        context_lines: Number of context lines around matches
        
    Returns:
        Dict with search results
    """
    
    try:
        from pathlib import Path
        import fnmatch
        
        if not os.path.exists(directory):
            return {"error": f"Directory '{directory}' does not exist"}
        
        search_term = search_term.lower()
        results = []
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # Check if file matches pattern
                if not fnmatch.fnmatch(file, file_pattern):
                    continue
                
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    matches = []
                    for i, line in enumerate(lines):
                        if search_term in line.lower():
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            
                            context = []
                            for j in range(start, end):
                                marker = ">>> " if j == i else "    "
                                context.append(f"{marker}{j+1:4d}: {lines[j].rstrip()}")
                            
                            matches.append({
                                "line_number": i + 1,
                                "line": line.strip(),
                                "context": context
                            })
                    
                    if matches:
                        results.append({
                            "file": file_path,
                            "matches": matches
                        })
                
                except Exception:
                    # Skip files that can't be read
                    continue
        
        return {
            "search_term": search_term,
            "directory": directory,
            "file_pattern": file_pattern,
            "results": results,
            "total_files_with_matches": len(results)
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict with file information
    """
    
    try:
        if not os.path.exists(file_path):
            return {"error": f"File '{file_path}' does not exist"}
        
        stat_info = os.stat(file_path)
        path_obj = Path(file_path)
        
        return {
            "path": file_path,
            "name": path_obj.name,
            "extension": path_obj.suffix,
            "size": stat_info.st_size,
            "is_directory": os.path.isdir(file_path),
            "is_file": os.path.isfile(file_path),
            "parent_directory": str(path_obj.parent),
            "exists": True
        }
        
    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}
