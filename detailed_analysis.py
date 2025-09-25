#!/usr/bin/env python3
import re
from collections import defaultdict
import sys

def analyze_imports(import_lines):
    """Analyze import statements and categorize them."""
    categories = {
        'standard_library': set(),
        'third_party': set(),
        'local': set()
    }
    
    standard_libraries = {
        'argparse', 'ast', 'collections', 'copy', 'datetime', 'enum', 'functools',
        'json', 'math', 'multiprocess', 'os', 'pathlib', 'random', 're', 'shutil',
        'subprocess', 'sys', 'time', 'typing', 'warnings', 'zipfile'
    }
    
    # Known third-party libraries
    third_party_libraries = {
        'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'cv2',
        'PIL', 'tqdm', 'faiss', 'sklearn', 'wandb', 'Cython', 'setuptools',
        'albumentations', 'requests', 'yaml', 'easydict', 'bisect', 'errno'
    }
    
    libraries = defaultdict(int)
    category_counts = defaultdict(int)
    
    for line in import_lines:
        line = line.strip()
        
        # Handle "import library" or "import library as alias"
        if line.startswith("import "):
            imports = line[7:].split(",")
            for imp in imports:
                imp = imp.strip()
                if " as " in imp:
                    imp = imp.split(" as ")[0].strip()
                base_lib = imp.split(".")[0].strip()
                if base_lib:
                    libraries[base_lib] += 1
                    
                    # Categorize
                    if base_lib in standard_libraries:
                        categories['standard_library'].add(base_lib)
                        category_counts['standard_library'] += 1
                    elif base_lib in third_party_libraries:
                        categories['third_party'].add(base_lib)
                        category_counts['third_party'] += 1
                    else:
                        categories['local'].add(base_lib)
                        category_counts['local'] += 1
        
        # Handle "from library import something"
        elif line.startswith("from "):
            match = re.match(r'from\s+([^(\s]+)\s+import', line)
            if match:
                lib = match.group(1)
                base_lib = lib.split(".")[0].strip()
                if base_lib:
                    libraries[base_lib] += 1
                    
                    # Categorize
                    if base_lib in standard_libraries:
                        categories['standard_library'].add(base_lib)
                        category_counts['standard_library'] += 1
                    elif base_lib in third_party_libraries:
                        categories['third_party'].add(base_lib)
                        category_counts['third_party'] += 1
                    else:
                        categories['local'].add(base_lib)
                        category_counts['local'] += 1
    
    return libraries, categories, category_counts

def main():
    # Read the import lines from stdin
    import_lines = []
    for line in sys.stdin:
        if line.strip():
            import_lines.append(line)
    
    libraries, categories, category_counts = analyze_imports(import_lines)
    
    # Sort by count (descending) then by library name (ascending)
    sorted_libraries = sorted(libraries.items(), key=lambda x: (-x[1], x[0]))
    
    print("Comprehensive Library Import Analysis for OpenUnReID Project")
    print("=" * 60)
    print(f"Total unique libraries: {len(sorted_libraries)}")
    print(f"Total import statements: {len(import_lines)}")
    print()
    
    # Print category breakdown
    print("Import Categories:")
    print("-" * 30)
    for category, count in category_counts.items():
        unique_libs = len(categories[category])
        print(f"{category.replace('_', ' ').title():<20} {count:>5} imports ({unique_libs} unique libs)")
    print()
    
    # Print detailed library usage
    print("Library Usage Details:")
    print("-" * 30)
    
    for lib, count in sorted_libraries:
        category = "Unknown"
        if lib in categories['standard_library']:
            category = "Standard"
        elif lib in categories['third_party']:
            category = "Third-party"
        elif lib in categories['local']:
            category = "Local"
        
        print(f"{lib:<25} {count:>5} ({category})")
    
    print()
    print("Essential Third-Party Libraries for ReID Project:")
    print("-" * 50)
    essential_libs = ['torch', 'torchvision', 'numpy', 'matplotlib', 'seaborn', 
                    'cv2', 'PIL', 'tqdm', 'faiss', 'sklearn', 'wandb', 'Cython']
    
    for lib in essential_libs:
        if lib in libraries:
            print(f"✓ {lib:<15} ({libraries[lib]} imports)")
        else:
            print(f"✗ {lib:<15} (not found)")

if __name__ == "__main__":
    main()