# sglang_patch_loader.py
'''
Usage: 
Use install_sglang_hook_patch.sh to install the SGLang patch.
Use uninstall_sglang_hook_patch.sh to remove the patch.
Or manually:
Put sglang_patch_loader.py & sglang_weight_hook_patch_core.py & sglang_injector.pth into the python site-packages directory of the target environment.
Use this command to find the site-packages directory:
python -c "import site; print(site.getsitepackages()[0])"
'''

import sys
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

import sglang_weight_hook_patch_core

TARGET_MODULES = {
    "sglang.srt.model_executor.model_runner"
}

_patch_applied = False

class SGLangPatcherLoader(Loader):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def exec_module(self, module):
        self.original_loader.exec_module(module)

        global _patch_applied
        if not _patch_applied:
            print(f"[SGLANG_PATCH_LOADER] Target module '{module.__name__}' loaded. Applying patches...")
            try:
                sglang_weight_hook_patch_core.apply_model_runner_patches()
                _patch_applied = True
                print(f"[SGLANG_PATCH_LOADER] Patches applied successfully in process {sglang_weight_hook_patch_core.os.getpid()}.")
            except Exception as e:
                print(f"[SGLANG_PATCH_LOADER] Error applying patches: {e}", file=sys.stderr)


class SGLangPatcherFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGET_MODULES or _patch_applied:
            return None

        original_finder = self
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec:
                spec.loader = SGLangPatcherLoader(spec.loader)
                return spec
        return None

sys.meta_path.insert(0, SGLangPatcherFinder())

print(f"[SGLANG_PATCH_LOADER] SGLang patch loader initialized in process {sglang_weight_hook_patch_core.os.getpid()}. Waiting for target module import...")
