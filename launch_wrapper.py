import sys
import os
import runpy

print(f"[WRAPPER] Starting main process with PID: {os.getpid()}.")

# --- 1. Set path ---
try:
    wrapper_dir = os.path.dirname(os.path.abspath(__file__))
    python_source_dir = os.path.join(wrapper_dir, "python")
    
    sys.path.insert(0, python_source_dir)
    sys.path.insert(0, wrapper_dir)
    print(f"[WRAPPER] sys.path configured. Added: {python_source_dir} and {wrapper_dir}")
except Exception as e:
    print(f"[WRAPPER] FATAL: Failed to configure sys.path: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. Entrypoint patch ---
try:
    import weight_hook_patch
    
    weight_hook_patch.apply_entrypoint_patches()

except Exception as e:
    print(f"[WRAPPER] FATAL: Failed to import or apply entrypoint patches: {e}", file=sys.stderr)
    sys.exit(1)

# --- 3. Launch server ---
if __name__ == "__main__":
    try:
        runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)
    except Exception as e:
        print(f"[WRAPPER] FATAL: An error occurred during SGLang server execution: {e}", file=sys.stderr)
        sys.exit(1)
