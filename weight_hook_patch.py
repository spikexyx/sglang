import sys
import os

# wrapper_dir = os.path.dirname(os.path.abspath(__file__))
# python_source_dir = os.path.join(wrapper_dir, "python")
# sys.path.insert(0, python_source_dir)

import fcntl
# import runpy
import json
import time
import torch
from typing import List, Tuple, Union, Optional
# import sglang.srt.model_executor.model_runner as model_runner_module

print(f"[SGLANG_PATCH] Patch Module loaded in process: {os.getpid()}")
# ===================================================================
# All patching code for model runner to handle weight metadata saving
# ===================================================================
def _patched_acquire_weight_lock(self, timeout=10):
    """acquire weight metadata saving file lock"""
    os.makedirs("weights_metadata", exist_ok=True)
    lock_file = os.path.join("weights_metadata", f"weight_saving_{self.gpu_id}.lock")

    try:
        self._lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY)
        start_time = time.time()

        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # logger.info(f"Acquired weight saving lock for GPU {self.gpu_id}")
                return True
            except IOError:
                if time.time() - start_time > timeout:
                    # logger.error(f"Failed to acquire weight lock within {timeout} seconds")
                    os.close(self._lock_fd)
                    return False
                time.sleep(0.1)
    except Exception as e:
        # logger.error(f"Error acquiring weight lock: {e}")
        return False

def _patched_release_weight_lock(self):
    """release weight metadata saving file lock"""
    if hasattr(self, '_lock_fd'):
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            # delete lock file
            lock_file = os.path.join("weights_metadata", f"weight_saving_{self.gpu_id}.lock")
            if os.path.exists(lock_file):
                os.remove(lock_file)
            # logger.info(f"Released weight saving lock for GPU {self.gpu_id}")
        # except Exception as e:
            # logger.warning(f"Error releasing weight lock: {e}")
        finally:
            delattr(self, '_lock_fd')

# Weights_hook function 
def _patched_register_weight_hooks(self):
    # self.weight_infos = {}  # Save weight metadatas
    self._clear_old_weight_data()

    def tensor_hook(tensor: torch.Tensor, name: str):
        if tensor.is_cuda:
            self.weight_infos[name] = {
                "ptr": tensor.data_ptr(),
                "size": tensor.numel() * tensor.element_size(),
                # "actual_size": tensor.storage().size() * tensor.element_size(),
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape)
            }

    if not self._acquire_weight_lock():
        raise RuntimeError("Failed to acquire weight metadata update lock")

    # Register hooks to capture the initial state of model weights
    for name, param in self.model.named_parameters():
        tensor_hook(param, name)  # Capture parameter weights
    self._save_weight_meta()  # Save weight metadata to a local file
    self.total_weight_dict = self._calculate_device_weight_sizes(unit="GB")
    self._save_total_weight_meta()
    # self._merge_weights()  # Merge weights based on pointer continuity
    # self._save_merged_weight_meta()  # Save merged weight metadata to a local file
    self._release_weight_lock()

# Save the model weight metadata to a JSON file
def _patched_save_weight_meta(self):
    os.makedirs("weights_metadata", exist_ok=True)
    meta_path = os.path.join("weights_metadata", f"weights_meta_{self.gpu_id}.json")
    # meta_path = f"weights_meta_{self.gpu_id}.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(self.weight_infos, f, indent=2)
        # logger.info(f"Save weight metadata to {meta_path}.")
    except IOError as e:
        # logger.error(f"Failed to save weight metadata to {meta_path}: {e}")
        raise

def _patched_save_total_weight_meta(self):
    os.makedirs("weights_metadata", exist_ok=True)
    meta_path = os.path.join("weights_metadata", f"total_weight_meta_{self.gpu_id}.json")
    # meta_path = f"weights_meta_{self.gpu_id}.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(self.total_weight_dict, f, indent=2)
        # logger.info(f"Save total weight metadata to {meta_path}.")
    except IOError as e:
        # logger.error(f"Failed to save total weight metadata to {meta_path}: {e}")
        raise

def _patched_calculate_device_weight_sizes(self, unit: str = "bytes") -> dict:
    """Calculate the total size of weights per device in self.weight_infos.
    
    Args:
        unit (str): The unit to return the size in. 
                    Options: "bytes", "KB", "MB", "GB".
    
    Returns:
        dict: {device: total_size} where total_size is in the specified unit.
    """
    device_sizes = {}  # {device: total_size_in_bytes}

    # 遍历所有 weight_infos，按 device 累加 size
    for info in self.weight_infos.values():
        device = info["device"]
        size = info["size"]
        if device in device_sizes:
            device_sizes[device] += size
        else:
            device_sizes[device] = size

    # 单位转换
    unit = unit.upper()
    if unit == "KB":
        return {device: size / 1024 for device, size in device_sizes.items()}
    elif unit == "MB":
        return {device: size / (1024 ** 2) for device, size in device_sizes.items()}
    elif unit == "GB":
        return {device: size / (1024 ** 3) for device, size in device_sizes.items()}
    else:  # Default to bytes
        return device_sizes
    
# Functions for recording weight metadata during inference when update model weights
def _patched_handle_weight_update_hooks(self):
    """
    Handle weight updates during inference - clean old data and capture new weight information
    """
    # logger.info("Starting weight update hook processing...")
    if not self._acquire_weight_lock():
        raise RuntimeError("Failed to acquire weight metadata update lock")

    # Clear old weight information
    self._clear_old_weight_data()

    # Re-register hooks to capture updated weights
    self._register_updated_weight_hooks()

    # Save updated metadata
    self._save_updated_weight_metadata()

    self._release_weight_lock()

    # logger.info("Weight update hook processing completed.")

def _patched_clear_old_weight_data(self):
    """
    Clear old weight information and metadata files
    """
    # Clear in-memory data
    if hasattr(self, 'weight_infos'):
        self.weight_infos.clear()
    else:
        self.weight_infos = {}

    if hasattr(self, 'total_weight_dict'):
        self.total_weight_dict.clear()
    else:
        self.total_weight_dict = {}

    # Remove old metadata files
    try:
        weights_dir = "weights_metadata"
        if os.path.exists(weights_dir):
            old_weight_file = os.path.join(weights_dir, f"weights_meta_{self.gpu_id}.json")
            old_total_file = os.path.join(weights_dir, f"total_weight_meta_{self.gpu_id}.json")

            if os.path.exists(old_weight_file):
                os.remove(old_weight_file)
                # logger.info(f"Removed old weight metadata file: {old_weight_file}")

            if os.path.exists(old_total_file):
                os.remove(old_total_file)
                # logger.info(f"Removed old total weight metadata file: {old_total_file}")

    except Exception as e:
        # logger.warning(f"Failed to clean old metadata files: {e}")
        return

def _patched_register_updated_weight_hooks(self):
    """
    Register hooks for updated model weights (similar to _register_weight_hooks but for updates)
    """
    def tensor_hook(tensor: torch.Tensor, name: str):
        if tensor.is_cuda:
            self.weight_infos[name] = {
                "ptr": tensor.data_ptr(),
                "size": tensor.numel() * tensor.element_size(),
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "updated": True  # Mark as updated weight
            }

    # Capture updated model weights
    # logger.info("Capturing updated model weights...")
    # weight_count = 0
    for name, param in self.model.named_parameters():
        tensor_hook(param, name)
        # weight_count += 1

    # logger.info(f"Captured {weight_count} updated weight tensors")

    # Calculate device weight sizes for updated weights
    self.total_weight_dict = self._calculate_device_weight_sizes(unit="GB")

def _patched_save_updated_weight_metadata(self):
    """
    Save updated weight metadata to JSON files
    """
    try:
        # Save individual weight metadata
        self._save_weight_meta()

        # Save total weight metadata
        self._save_total_weight_meta()

        # Additionally save update timestamp and summary
        self._save_weight_update_summary()

    except Exception as e:
        # logger.error(f"Failed to save updated weight metadata: {e}")
        return

def _patched_save_weight_update_summary(self):
    """
    Save a summary of the weight update operation
    """
    import time

    summary = {
        "update_timestamp": time.time(),
        "update_time_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "gpu_id": self.gpu_id,
        "total_weights": len(self.weight_infos),
        "total_devices": len(self.total_weight_dict),
        "device_weight_summary": self.total_weight_dict,
        "memory_usage_gb": sum(self.total_weight_dict.values()) if self.total_weight_dict else 0
    }

    os.makedirs("weights_metadata", exist_ok=True)
    summary_path = os.path.join("weights_metadata", f"weight_update_summary_{self.gpu_id}.json")

    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        # logger.info(f"Saved weight update summary to {summary_path}")
    except IOError as e:
        # logger.error(f"Failed to save weight update summary to {summary_path}: {e}")
        return

def _patched_validate_weight_update(self):
    """
    Validate that weight update was successful by checking if weights have new pointers
    """
    if not self.weight_infos:
        # logger.warning("No weight information found after update")
        return False

    # Check if we have the expected number of weights
    expected_weight_count = sum(1 for _ in self.model.named_parameters())
    actual_weight_count = len(self.weight_infos)

    if actual_weight_count != expected_weight_count:
        # logger.warning(f"Weight count mismatch: expected {expected_weight_count}, got {actual_weight_count}")
        return False

    # Check if all weights are marked as CUDA tensors
    cuda_weights = sum(1 for info in self.weight_infos.values() if "cuda" in info["device"])
    if cuda_weights == 0:
        # logger.warning("No CUDA weights found after update")
        return False

    # logger.info(f"Weight update validation passed: {actual_weight_count} weights, {cuda_weights} on CUDA")
    return True

# Entry hook for updating weights metadata
def _patched_update_weights_metadata(self):
    """
    Public interface to update weight metadata
    """
    try:
        self._handle_weight_update_hooks()

        # Validate the update
        if self._validate_weight_update():
            # logger.info("Weight metadata update completed successfully")
            return True
        else:
            # logger.error("Weight metadata update validation failed")
            return False

    except Exception as e:
        # logger.error(f"Weight metadata update failed: {e}")
        return False
# ===================================================================
# print("[PATCH] All patches have been applied.")

# ===================================================================
# Monkey patch the ModelRunner class methods

def apply_model_runner_patches():
    print(f"[PATCH] Applying model runner patches in process {os.getpid()}...")
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner

        ModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        ModelRunner._release_weight_lock = _patched_release_weight_lock
        ModelRunner._register_weight_hooks = _patched_register_weight_hooks
        ModelRunner._save_weight_meta = _patched_save_weight_meta
        ModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        ModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        ModelRunner._handle_weight_update_hooks = _patched_handle_weight_update_hooks
        ModelRunner._clear_old_weight_data = _patched_clear_old_weight_data
        ModelRunner._register_updated_weight_hooks = _patched_register_updated_weight_hooks
        ModelRunner._save_updated_weight_metadata = _patched_save_updated_weight_metadata
        ModelRunner._save_weight_update_summary = _patched_save_weight_update_summary
        ModelRunner._validate_weight_update = _patched_validate_weight_update
        ModelRunner.update_weights_metadata = _patched_update_weights_metadata

        # Patch load_model
        original_load_model = ModelRunner.load_model

        def patched_load_model(self):
            print("[PATCH] Patching ModelRunner.load_model to handle weight metadata loading")
            original_load_model(self)
            # Register hooks after model is loaded
            self._register_weight_hooks()

        ModelRunner.load_model = patched_load_model

        # Patch update_weights_from_disk
        original_update_weights_from_disk = ModelRunner.update_weights_from_disk

        def patched_update_weights_from_disk(
                self, model_path: str, load_format: str
            ) -> tuple[bool, str]:
            print("[PATCH] Patching ModelRunner.update_weights_from_disk to handle update weight metadata loading")
            result = original_update_weights_from_disk(self, model_path, load_format)
            # Register hooks after weights are updated
            self.update_weights_metadata()
            return result
        
        ModelRunner.update_weights_from_disk = patched_update_weights_from_disk
        
        # Patch update_weights_from_tensor
        original_update_weights_from_tensor = ModelRunner.update_weights_from_tensor

        def patched_update_weights_from_tensor(
                self,
                named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
                load_format: Optional[str] = None,
            ):
            print("[PATCH] Patching ModelRunner.update_weights_from_tensor to handle update weight metadata loading")
            result = original_update_weights_from_tensor(self, named_tensors, load_format)
            # Register hooks after weights are updated
            self.update_weights_metadata()
            return result
        
        ModelRunner.update_weights_from_tensor = patched_update_weights_from_tensor

    except Exception as e:
        print(f"[PATCH] Failed to apply ModelRunner patches: {e}")
        raise

# ====================================================================
# Patch the run_scheduler_process and run_data_parallel_controller_process functions (subprocesses)
from sglang.srt.server_args import ServerArgs, PortArgs

original_run_scheduler_process = None
original_run_data_parallel_controller_process = None

def patched_run_scheduler_process(
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        pipe_writer,
    ):
    print(f"[PATCH] Patching run_scheduler_process for GPU {gpu_id}, TP rank {tp_rank}, PP rank {pp_rank}, DP rank {dp_rank} in process {os.getpid()} ...")
    apply_model_runner_patches()

    if original_run_scheduler_process:
        original_run_scheduler_process(
            server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank, pipe_writer
        )
    else:
        print("[PATCH] No original run_scheduler_process found, skipping.")

def patched_run_data_parallel_controller_process(
        server_args: ServerArgs,
        port_args: PortArgs,
        pipe_writer,
    ):
    print(f"[PATCH] Patching run_data_parallel_controller_process in process {os.getpid()} ...")
    apply_model_runner_patches()

    if original_run_data_parallel_controller_process:
        original_run_data_parallel_controller_process(server_args, port_args, pipe_writer)
    else:
        print("[PATCH] No original run_data_parallel_controller_process found, skipping.")
    
# ===================================================================
def apply_entrypoint_patches():
    print(f"[PATCH] Applying entrypoint patches for SGLang server in {os.getpid()} ...")
    global original_run_scheduler_process, original_run_data_parallel_controller_process

    try:
        from sglang.srt.managers.scheduler import run_scheduler_process
        from sglang.srt.managers.data_parallel_controller import run_data_parallel_controller_process

        original_run_scheduler_process = run_scheduler_process
        original_run_data_parallel_controller_process = run_data_parallel_controller_process

        # Patch the functions
        run_scheduler_process = patched_run_scheduler_process
        run_data_parallel_controller_process = patched_run_data_parallel_controller_process

    except Exception as e:
        print(f"[PATCH] Failed to import necessary modules for entrypoint patching: {e}")
        raise

# ===================================================================
# Patch the load_model method to handle weight metadata loading
# original_load_model = model_runner_module.ModelRunner.load_model

# def patched_load_model(self):
#     print("[PATCH] Patching ModelRunner.load_model to handle weight metadata loading")
#     original_load_model(self)
#     # Register hooks after model is loaded
#     self._register_weight_hooks()  

# ===================================================================
# Patch the update_weights_from_disk method to handle update weight
# original_update_weights_from_disk = model_runner_module.ModelRunner.update_weights_from_disk

# def patched_update_weights_from_disk(
#         self, model_path: str, load_format: str
#     ) -> tuple[bool, str]:
#     print("[PATCH] Patching ModelRunner.update_weights_from_disk to handle update weight metadata loading")
#     result = original_update_weights_from_disk(self, model_path, load_format)
#     # Register hooks after weights are updated
#     self.update_weights_metadata()
#     return result

# ===================================================================
# Patch the update_weights_from_tensor method to handle update weight
# original_update_weights_from_tensor = model_runner_module.ModelRunner.update_weights_from_tensor

# def patched_update_weights_from_tensor(
#         self,
#         named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
#         load_format: Optional[str] = None,
#     ):
#     print("[PATCH] Patching ModelRunner.update_weights_from_tensor to handle update weight metadata loading")
#     result = original_update_weights_from_tensor(self, named_tensors, load_format)
#     # Register hooks after weights are updated
#     self.update_weights_metadata()
#     return result

# ===================================================================
# Apply the patches to the ModelRunner class
# model_runner_module.ModelRunner.load_model = patched_load_model
# model_runner_module.ModelRunner.update_weights_from_disk = patched_update_weights_from_disk
# model_runner_module.ModelRunner.update_weights_from_tensor = patched_update_weights_from_tensor
