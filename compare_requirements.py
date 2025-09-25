#!/usr/bin/env python3
"""
Comparison script to show the difference between original and cleaned requirements.txt
"""

def compare_requirements():
    # Read original requirements
    with open('requirements_backup.txt', 'r') as f:
        original_lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    
    # Read cleaned requirements
    with open('requirements.txt', 'r') as f:
        cleaned_lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    
    original_packages = set()
    cleaned_packages = set()
    
    for line in original_lines:
        if '==' in line:
            pkg = line.split('==')[0]
        elif '>=' in line:
            pkg = line.split('>=')[0]
        else:
            pkg = line
        original_packages.add(pkg.lower())
    
    for line in cleaned_lines:
        if '==' in line:
            pkg = line.split('==')[0]
        elif '>=' in line:
            pkg = line.split('>=')[0]
        else:
            pkg = line
        cleaned_packages.add(pkg.lower())
    
    removed_packages = original_packages - cleaned_packages
    kept_packages = cleaned_packages & original_packages
    new_packages = cleaned_packages - original_packages
    
    print("Requirements.txt Comparison Report")
    print("=" * 50)
    print()
    
    print(f"Original packages: {len(original_packages)}")
    print(f"Cleaned packages: {len(cleaned_packages)}")
    print(f"Removed packages: {len(removed_packages)}")
    print(f"Kept packages: {len(kept_packages)}")
    print(f"New packages: {len(new_packages)}")
    print()
    
    print(f"Reduction: {len(removed_packages)}/{len(original_packages)} packages ({len(removed_packages)/len(original_packages)*100:.1f}%)")
    print()
    
    print("✅ KEPT PACKAGES:")
    print("-" * 30)
    for pkg in sorted(kept_packages):
        print(f"  • {pkg}")
    
    print()
    print("❌ REMOVED PACKAGES:")
    print("-" * 30)
    for pkg in sorted(removed_packages):
        print(f"  • {pkg}")
    
    print()
    print("➕ NEW PACKAGES:")
    print("-" * 30)
    for pkg in sorted(new_packages):
        print(f"  • {pkg}")
    
    print()
    print("💾 FILES CREATED:")
    print("-" * 30)
    print("  • requirements_backup.txt - Original requirements")
    print("  • requirements.txt - Cleaned requirements")
    print("  • requirements_clean.txt - Copy of cleaned requirements")
    
    print()
    print("⚠️  IMPORTANT NOTES:")
    print("-" * 30)
    print("• Some removed packages may be transitive dependencies")
    print("• Test the project thoroughly before committing changes")
    print("• Consider using pip freeze > requirements.txt to capture actual dependencies")
    print("• Some packages may be needed for specific features not yet analyzed")

if __name__ == "__main__":
    compare_requirements()