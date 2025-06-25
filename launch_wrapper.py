import sys
import os

wrapper_dir = os.path.dirname(os.path.abspath(__file__))
python_source_dir = os.path.join(wrapper_dir, "python")
sys.path.insert(0, python_source_dir)
sys.path.insert(0, wrapper_dir)

import fcntl
import runpy
import json
import time
import torch
from typing import List, Tuple, Union, Optional

from sglang.srt.server_args import ServerArgs
from sglang.srt.server_args import PortArgs

print(f"[WRAPPER] Preparing to patch the scheduler process in {os.getpid()}...")

# tp situation
try:
    import sglang.srt.managers.scheduler as scheduler_module

    original_run_scheduler_process = scheduler_module.run_scheduler_process

    def patched_run_scheduler_process(
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        pipe_writer,
    ):
        print(f"[WRAPPER] Patching the scheduler process in {os.getpid()}...")
        import weight_hook_patch
        # Call the original function
        original_run_scheduler_process(
            server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank, pipe_writer
        )
        print(f"[WRAPPER] Scheduler process patched successfully in {os.getpid()}.")

    scheduler_module.run_scheduler_process = patched_run_scheduler_process
    print(f"[WRAPPER] Scheduler process patch applied in run_scheduler_process in {os.getpid()}.")

except ImportError as e:
    print(f"[WRAPPER] Failed to patch the scheduler process: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[WRAPPER] An error occurred while patching the scheduler process: {e}")
    sys.exit(1)


# dp situation
try:
    import sglang.srt.managers.data_parallel_controller as dp_controller_module

    original_run_data_parallel_controller_process = dp_controller_module.run_data_parallel_controller_process

    def patched_run_data_parallel_controller_process (
        server_args: ServerArgs,
        port_args: PortArgs,
        pipe_writer,
    ):
        print(f"[WRAPPER] Patching the data parallel controller process in {os.getpid()}...")
        import weight_hook_patch
        # Call the original function
        original_run_data_parallel_controller_process(server_args, port_args, pipe_writer)
        print(f"[WRAPPER] Data parallel controller process patched successfully in {os.getpid()}.")

    dp_controller_module.run_data_parallel_controller_process = patched_run_data_parallel_controller_process
    print(f"[WRAPPER] Data parallel controller process patch applied in run_data_parallel_controller_process in {os.getpid()}.")

except ImportError as e:
    print(f"[WRAPPER] Failed to patch the data parallel controller process: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[WRAPPER] An error occurred while patching the data parallel controller process: {e}")
    sys.exit(1)

# ===================================================================
# Apply the patches and run
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)