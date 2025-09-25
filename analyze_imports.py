#!/usr/bin/env python3
import re
from collections import defaultdict
import sys

def extract_libraries_from_imports(import_lines):
    """Extract library names from import statements."""
    libraries = defaultdict(int)
    
    for line in import_lines:
        line = line.strip()
        
        # Handle "import library" or "import library as alias"
        if line.startswith("import "):
            # Remove "import " and split by comma for multiple imports
            imports = line[7:].split(",")
            for imp in imports:
                imp = imp.strip()
                # Remove any "as alias" part
                if " as " in imp:
                    imp = imp.split(" as ")[0].strip()
                # Get the base library name (before any dots)
                base_lib = imp.split(".")[0].strip()
                if base_lib:
                    libraries[base_lib] += 1
        
        # Handle "from library import something"
        elif line.startswith("from "):
            # Extract the library name between "from " and " import"
            match = re.match(r'from\s+([^(\s]+)\s+import', line)
            if match:
                lib = match.group(1)
                # Get the base library name (before any dots)
                base_lib = lib.split(".")[0].strip()
                if base_lib:
                    libraries[base_lib] += 1
    
    return libraries

def main():
    # Read the import lines from stdin
    import_lines = []
    for line in sys.stdin:
        if line.strip():
            import_lines.append(line)
    
    libraries = extract_libraries_from_imports(import_lines)
    
    # Sort by count (descending) then by library name (ascending)
    sorted_libraries = sorted(libraries.items(), key=lambda x: (-x[1], x[0]))
    
    print("Library Import Analysis for OpenUnReID Project")
    print("=" * 50)
    print(f"Total unique libraries: {len(sorted_libraries)}")
    print(f"Total import statements: {len(import_lines)}")
    print()
    print("Library usage count:")
    print("-" * 30)
    
    for lib, count in sorted_libraries:
        print(f"{lib:<25} {count:>5}")

if __name__ == "__main__":
    main()