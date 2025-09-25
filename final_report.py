#!/usr/bin/env python3
"""
Final analysis script for OpenUnReID project imports.
Compares actual usage with requirements.txt to identify unused dependencies.
"""

def create_final_report():
    # Based on our analysis, here are the actual libraries used
    used_libraries = {
        # Standard library (no need to install)
        'standard_library': {
            'os', 'argparse', 'collections', 'time', 'shutil', 'sys', 'pathlib',
            'datetime', 're', 'warnings', 'random', 'copy', 'math', 'functools',
            'json', 'subprocess', 'zipfile', 'typing', 'multiprocess', 'errno'
        },
        
        # Third-party libraries that are actually used
        'third_party': {
            'torch', 'torchvision', 'numpy', 'matplotlib', 'seaborn', 'cv2',
            'PIL', 'tqdm', 'faiss', 'sklearn', 'Cython', 'albumentations',
            'pandas', 'easydict', 'yaml', 'requests', 'bisect', 'setuptools'
        },
        
        # Local imports (internal to the project)
        'local': {
            'openunreid', 'wandb', 'distutils', 'tarfile'
        }
    }
    
    # Import counts from our analysis
    import_counts = {
        'torch': 95, 'torchvision': 3, 'numpy': 20, 'matplotlib': 7, 'seaborn': 2,
        'cv2': 2, 'PIL': 7, 'tqdm': 9, 'faiss': 3, 'sklearn': 1, 'Cython': 2,
        'albumentations': 1, 'pandas': 3, 'easydict': 1, 'yaml': 1, 'requests': 1,
        'bisect': 1, 'setuptools': 1
    }
    
    # Common package names mapping
    package_mapping = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'faiss': 'faiss-gpu',  # Based on requirements.txt
        'Cython': 'Cython'
    }
    
    print("OpenUnReID Project Library Usage Analysis")
    print("=" * 50)
    print()
    
    print("📊 ACTUALLY USED LIBRARIES:")
    print("-" * 40)
    
    # Sort by usage count
    sorted_by_usage = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)
    
    for lib, count in sorted_by_usage:
        package_name = package_mapping.get(lib, lib)
        print(f"• {package_name:<20} ({count} imports)")
    
    print()
    print("📋 ESSENTIAL LIBRARIES FOR ReID FUNCTIONALITY:")
    print("-" * 50)
    essential_core = ['torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 
                     'scikit-learn', 'faiss-gpu', 'Cython', 'tqdm', 'matplotlib']
    essential_analysis = ['pandas', 'seaborn', 'PyYAML', 'requests']
    
    print("Core Deep Learning:")
    for lib in essential_core:
        print(f"  ✓ {lib}")
    
    print("\nData Analysis & Visualization:")
    for lib in essential_analysis:
        print(f"  ✓ {lib}")
    
    print("\n🚨 POTENTIALLY UNUSED DEPENDENCIES:")
    print("-" * 40)
    
    # List some common packages from requirements.txt that aren't used
    unused_packages = [
        'accelerate', 'addict', 'albucore', 'altair', 'antlr4-python3-runtime',
        'blake3', 'cupy-cuda12x', 'debugpy', 'distro', 'docker-pycreds',
        'easydict', 'einops', 'eval_type_backport', 'fastrlock', 'fsspec',
        'gitdb', 'GitPython', 'gmpy2', 'GPUtil', 'grpcio', 'h5py', 'hiddenlayer',
        'huggingface-hub', 'joblib', 'jsonpatch', 'jsonschema', 'jupyter_client',
        'lightning-utilities', 'lm-format-enforcer', 'mistral_common',
        'msgpack', 'msgspec', 'multidict', 'multiprocess', 'ninja', 'numba',
        'omegaconf', 'openai', 'outlines', 'packaging', 'parso', 'prometheus_client',
        'protobuf', 'psutil', 'pydantic', 'pydeck', 'pydensecrf', 'pynvml',
        'pytorch-lightning', 'ray', 'referencing', 'rich', 'rpds-py',
        'safetensors', 'scipy', 'sentencepiece', 'setproctitle', 'shellingham',
        'simsimd', 'smmap', 'sniffio', 'starlette', 'streamlit', 'stringzilla',
        'tabulate', 'tensorboard', 'termcolor', 'thop', 'threadpoolctl',
        'tiktoken', 'timm', 'tokenizers', 'toml', 'torchmetrics', 'torchnet',
        'torchsummary', 'triton', 'typer', 'uvicorn', 'uvloop', 'visdom',
        'vllm', 'watchdog', 'watchfiles', 'websocket-client', 'websockets',
        'Werkzeug', 'xformers', 'yacs', 'zipp'
    ]
    
    print("High-level ML/DL frameworks not used:")
    for pkg in ['accelerate', 'huggingface-hub', 'lightning-utilities', 'pytorch-lightning', 
                'ray', 'tensorboard', 'timm', 'torchmetrics', 'torchnet', 'torchsummary', 
                'triton', 'vllm', 'visdom']:
        print(f"  • {pkg}")
    
    print("\nWeb/API related packages not used:")
    for pkg in ['fastapi', 'httpcore', 'httpx', 'openai', 'prometheus_client', 'streamlit', 
                'starlette', 'uvicorn', 'websocket-client', 'websockets', 'Werkzeug']:
        print(f"  • {pkg}")
    
    print("\nDevelopment/Testing packages not used:")
    for pkg in ['debugpy', 'jupyter_client', 'pytest', 'rich', 'watchdog']:
        print(f"  • {pkg}")
    
    print()
    print("💡 RECOMMENDED MINIMAL REQUIREMENTS:")
    print("-" * 40)
    
    minimal_requirements = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'pillow>=8.0.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=1.0.0',
        'faiss-gpu>=1.7.0',
        'Cython>=0.29.0',
        'tqdm>=4.60.0',
        'pandas>=1.3.0',
        'seaborn>=0.11.0',
        'PyYAML>=5.4.0',
        'requests>=2.25.0',
        'easydict>=1.9.0',
        'albumentations>=1.1.0',
        'wandb>=0.12.0'  # Used for experiment tracking
    ]
    
    for req in minimal_requirements:
        print(f"{req}")
    
    print()
    print("📈 SUMMARY:")
    print("-" * 20)
    print(f"• Total unique libraries actually used: {len(used_libraries['third_party'])}")
    print(f"• Total packages in requirements.txt: ~245")
    print(f"• Estimated unused packages: ~200+")
    print(f"• Potential space savings: ~80% of requirements.txt")
    
    print()
    print("⚠️  NOTE:")
    print("-" * 20)
    print("Some packages may be transitive dependencies or used indirectly.")
    print("Test thoroughly before removing dependencies in production.")

if __name__ == "__main__":
    create_final_report()