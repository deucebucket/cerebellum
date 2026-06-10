import sys
import pkgutil
import inspect
import importlib
import os
from tqdm import tqdm

def get_stdlib_symbols():
    print("[*] Walking Python Standard Library Recursively...")
    
    symbols = []
    seen_symbols = set()
    
    # Standard library paths
    stdlib_paths = [os.path.dirname(os.__file__)]
    
    # Recursive walk
    for _, mod_name, is_pkg in pkgutil.walk_packages(path=stdlib_paths):
        if mod_name.startswith('_'): continue
        
        try:
            mod = importlib.import_module(mod_name)
            for name, obj in inspect.getmembers(mod):
                if name.startswith('_'): continue
                
                full_name = f"{mod_name}.{name}"
                if full_name in seen_symbols: continue
                
                # Capture all attributes with docstrings
                doc = inspect.getdoc(obj)
                if doc:
                    symbols.append({"symbol": full_name, "doc": doc})
                    seen_symbols.add(full_name)
        except:
            continue
            
    # Add builtins
    for name in dir(__builtins__):
        if name.startswith('_'): continue
        doc = inspect.getdoc(getattr(__builtins__, name))
        if doc:
            symbols.append({"symbol": name, "doc": doc})
            
    return symbols

if __name__ == "__main__":
    symbols = get_stdlib_symbols()
    print(f"[+] Successfully pulled {len(symbols)} symbols with documentation.")
    
    # Save to a new 13k corpus file
    with open("python_stdlib_13k.txt", "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(f"# {s['symbol']}\n{s['doc']}\n\n")
            
    print(f"[+] Saved to python_stdlib_13k.txt")
