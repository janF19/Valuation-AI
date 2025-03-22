import os
from pathlib import Path

def display_project_structure(start_path: str, indent: str = '    ', ignore_patterns: list = None):
    """
    Display the directory structure of a project.
    
    Args:
        start_path (str): Root directory to start from
        indent (str): Indentation string (default: 4 spaces)
        ignore_patterns (list): List of patterns to ignore (e.g., ['.git', '__pycache__', 'venv'])
    """
    if ignore_patterns is None:
        ignore_patterns = ['.git', '__pycache__', 'venv', 'node_modules', '.pytest_cache']

    def should_ignore(path):
        return any(pattern in str(path) for pattern in ignore_patterns)

    start_path = Path(start_path)
    
    print(f"\n📁 Project Structure for: {start_path.absolute()}\n")
    
    for path in sorted(start_path.rglob('*')):
        if should_ignore(path):
            continue
            
        depth = len(path.relative_to(start_path).parts)
        if path.is_file():
            print(f"{indent * (depth-1)}📄 {path.name}")
        elif path.is_dir() and depth > 0:  # Skip root directory
            print(f"{indent * (depth-1)}📁 {path.name}")

if __name__ == "__main__":
    # Use it from current directory
    display_project_structure('.')