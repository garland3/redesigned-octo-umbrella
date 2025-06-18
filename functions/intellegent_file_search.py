"""
Intelligent file search functionality.
Provides keyword-based search across directories and files.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def intelligent_file_search(
    search_term: str, 
    directory: str = ".", 
    file_extensions: Optional[List[str]] = None,
    max_results: int = 10,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search for files and content using keywords.
    Simple keyword-based search suitable for less powerful LLMs.
    
    Args:
        search_term: The term to search for (in filenames and content)
        directory: Directory to search in (default: current directory)
        file_extensions: List of file extensions to include (e.g., ['.py', '.txt'])
        max_results: Maximum number of results to return
        include_content: Whether to search inside file content
        
    Returns:
        Dict with search results
    """
    
    try:
        # Validate inputs
        if not search_term or len(search_term.strip()) == 0:
            return {"error": "Search term cannot be empty"}
        
        if not os.path.exists(directory):
            return {"error": f"Directory '{directory}' does not exist"}
        
        search_term = search_term.lower().strip()
        results = {
            "search_term": search_term,
            "directory": directory,
            "filename_matches": [],
            "content_matches": [],
            "total_files_searched": 0
        }
        
        # Default file extensions if none provided
        if file_extensions is None:
            file_extensions = ['.py', '.txt', '.md', '.rst', '.json', '.yaml', '.yml', '.log']
        
        # Search through directory
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                # Check if file extension matches
                if file_extensions and file_ext not in file_extensions:
                    continue
                
                results["total_files_searched"] += 1
                
                # Check filename match
                if search_term in file.lower():
                    results["filename_matches"].append({
                        "file": file_path,
                        "filename": file,
                        "match_type": "filename"
                    })
                
                # Search file content if requested
                if include_content and len(results["content_matches"]) < max_results:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if search_term in content.lower():
                                # Find context around the match
                                lines = content.split('\n')
                                matching_lines = []
                                
                                for i, line in enumerate(lines):
                                    if search_term in line.lower():
                                        # Get some context around the match
                                        start = max(0, i - 2)
                                        end = min(len(lines), i + 3)
                                        context = lines[start:end]
                                        
                                        matching_lines.append({
                                            "line_number": i + 1,
                                            "line": line.strip(),
                                            "context": context
                                        })
                                        
                                        # Limit matches per file
                                        if len(matching_lines) >= 3:
                                            break
                                
                                if matching_lines:
                                    results["content_matches"].append({
                                        "file": file_path,
                                        "filename": file,
                                        "matches": matching_lines[:3],  # Limit to 3 matches per file
                                        "match_type": "content"
                                    })
                    
                    except Exception as e:
                        # Skip files that can't be read
                        pass
                
                # Stop if we have enough results
                if (len(results["filename_matches"]) + len(results["content_matches"])) >= max_results:
                    break
        
        # Summary
        results["summary"] = {
            "total_filename_matches": len(results["filename_matches"]),
            "total_content_matches": len(results["content_matches"]),
            "files_searched": results["total_files_searched"]
        }
        
        return results
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def search_in_specific_file(file_path: str, search_term: str, context_lines: int = 2) -> Dict[str, Any]:
    """
    Search for a term in a specific file.
    
    Args:
        file_path: Path to the file to search
        search_term: Term to search for
        context_lines: Number of context lines around each match
        
    Returns:
        Dict with search results
    """
    
    try:
        if not os.path.exists(file_path):
            return {"error": f"File '{file_path}' does not exist"}
        
        search_term = search_term.lower().strip()
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        matches = []
        for i, line in enumerate(lines):
            if search_term in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                context = []
                for j in range(start, end):
                    prefix = ">>> " if j == i else "    "
                    context.append(f"{prefix}{j+1:4d}: {lines[j].rstrip()}")
                
                matches.append({
                    "line_number": i + 1,
                    "line": line.strip(),
                    "context": context
                })
        
        return {
            "file": file_path,
            "search_term": search_term,
            "matches": matches,
            "total_matches": len(matches)
        }
        
    except Exception as e:
        return {"error": f"Search in file failed: {str(e)}"}


def find_files_by_pattern(pattern: str, directory: str = ".", max_results: int = 20) -> Dict[str, Any]:
    """
    Find files matching a pattern (simple wildcard support).
    
    Args:
        pattern: Pattern to match (supports * and ?)
        directory: Directory to search
        max_results: Maximum number of results
        
    Returns:
        Dict with matching files
    """
    
    try:
        if not os.path.exists(directory):
            return {"error": f"Directory '{directory}' does not exist"}
        
        # Convert simple wildcards to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        regex_pattern = f"^{regex_pattern}$"
        
        matches = []
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if re.match(regex_pattern, file, re.IGNORECASE):
                    file_path = os.path.join(root, file)
                    matches.append({
                        "file": file_path,
                        "filename": file,
                        "directory": root
                    })
                    
                    if len(matches) >= max_results:
                        break
            
            if len(matches) >= max_results:
                break
        
        return {
            "pattern": pattern,
            "directory": directory,
            "matches": matches,
            "total_found": len(matches)
        }
        
    except Exception as e:
        return {"error": f"Pattern search failed: {str(e)}"}
