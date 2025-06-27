# sglang_patch_loader.py (Optimized: Patch only in the main process)

import os
import sys

def is_main_sglang_process():
    """
    Detects if the current process is the main SGLang server process.
    This combines the most reliable checks we've found.
    """
    argv = sys.argv
    main_module = sys.modules.get('__main__')

    # --- Check 1: The most reliable method via runpy ---
    # `python -m sglang.launch_server` is executed by the `runpy` module.
    if main_module and hasattr(main_module, '__file__') and main_module.__file__:
        if 'runpy.py' in main_module.__file__ and 'sglang.launch_server' in argv:
            print("[SGLANG_PATCH_LOADER] >> Detected main process via runpy.")
            return True

    # --- Check 2: Fallback for your specific environment's argv structure ---
    # sys.argv is ['-m', '--model-path', ...]
    if len(argv) > 1 and argv[0] == '-m' and '--model-path' in argv:
        print("[SGLANG_PATCH_LOADER] >> Detected main process via special argv heuristic.")
        return True
        
    # --- Check 3: Standard argv check (less likely for you but good for general use) ---
    if len(argv) > 0 and 'sglang/launch_server.py' in argv[0]:
        print("[SGLANG_PATCH_LOADER] >> Detected main process via script path in argv.")
        return True

    return False

def run_patch():
    """
    Applies the patch ONLY if the current process is identified as the
    main SGLang server process. Child processes will inherit the patched state
    and will not re-apply the patch.
    """
    # This entire block of code will run in every process (main and children)
    # because of the .pth mechanism.
    
    # However, we only take action in the main process.
    if is_main_sglang_process():
        print(f"[SGLANG_PATCH_LOADER] >> Main SGLang process (PID: {os.getpid()}) confirmed. Applying patch now...")
        try:
            # Import the patch core only when needed
            from sglang_weight_hook_patch_core import apply_entrypoint_patches
            
            apply_entrypoint_patches()
            
            print(f"[SGLANG_PATCH_LOADER] >> Patch successfully applied once in the main process.")
            # We add a sentinel to prevent re-patching even in the same process, just in case.
            # This is a good defensive practice.
            setattr(sys, '_sglang_patch_applied', True)

        except Exception as e:
            print(f"[SGLANG_PATCH_LOADER] !! ERROR: Failed to apply patch in the main process (PID: {os.getpid()}): {e}")
    
    # For child processes or any other python process, this script will now do nothing and exit silently.
    # You can add a print here for debugging if you want to see it running in children.
    # else:
    #     print(f"[SGLANG_PATCH_LOADER] >> Skipping patch in process {os.getpid()} (not main sglang process).")


# --- Entry point called by the .pth file ---
# Add a global guard to ensure this runs only once per process, even if imported multiple times.
if not hasattr(sys, '_sglang_patch_applied'):
    run_patch()

