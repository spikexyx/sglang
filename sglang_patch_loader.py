# sglang_patch_loader.py
'''
Usage: Put sglang_patch_loader.py & sglang_weight_hook_patch_core.py & sglang_injector.pth into the python site-packages directory of the target environment.
Use this command to find the site-packages directory:
python -c "import site; print(site.getsitepackages()[0])"
'''

import sys
import os

def apply_sglang_patches():
    """
    Checks if the target module is being run and applies patches if so.
    """
    # --- Safe check：Only apply patch when sglang.launch_server ---
    
    # Check `-m sglang.launch_server` feature
    is_launch_server = False
    if len(sys.argv) > 1 and sys.argv[1] == "-m" and "sglang.launch_server" in sys.argv:
         is_launch_server = True
    # Check `python sglang/launch_server.py` feature
    elif "sglang/launch_server.py" in sys.argv[0]:
         is_launch_server = True
    # Check "sglang.launch_server"
    if "sglang.launch_server" in " ".join(sys.argv):
        is_launch_server = True

    if not is_launch_server:
        return

    print("[SGLANG_PATCH_LOADER] Detected sglang.launch_server startup. Applying patches...")
    
    try:
        # sglang_weight_hook_patch_core.py and sglang_patch_loader.py should be in the same directory
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        if loader_dir not in sys.path:
            sys.path.insert(0, loader_dir)
            
        import sglang_weight_hook_patch_core
        sglang_weight_hook_patch_core.apply_entrypoint_patches()
        
        print("[SGLANG_PATCH_LOADER] Patches applied successfully.")

    except Exception as e:
        print(f"[SGLANG_PATCH_LOADER] WARNING: Failed to apply patches: {e}", file=sys.stderr)


apply_sglang_patches()