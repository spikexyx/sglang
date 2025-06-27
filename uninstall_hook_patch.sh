#!/bin/bash

# uninstall_patch.sh: A script to uninstall the SGLang non-intrusive patch.

# --- Configuration ---
PATCH_CORE_FILE="sglang_weight_hook_patch_core.py"
PATCH_LOADER_FILE="sglang_patch_loader.py"
PTH_FILE="sglang_injector.pth"

# --- Style Definitions ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

set -e

echo -e "${YELLOW}Starting SGLang Patch Uninstallation...${NC}"

# --- 1. Find the active Python environment's site-packages directory ---
echo "Detecting active Python environment..."
PYTHON_EXEC=$(command -v python3 || command -v python)

if [ -z "$PYTHON_EXEC" ]; then
    echo -e "${RED}Error: Could not find 'python' or 'python3' in your PATH.${NC}"
    exit 1
fi
echo "Using Python executable: ${PYTHON_EXEC}"

SITE_PACKAGES_DIR=$($PYTHON_EXEC -c "import site; print(site.getsitepackages()[0] if site.getsitepackages() else '')")

if [ -z "$SITE_PACKAGES_DIR" ] || [ ! -d "$SITE_PACKAGES_DIR" ]; then
    echo -e "${RED}Error: Could not determine a valid site-packages directory.${NC}"
    exit 1
fi
echo "Target site-packages directory: ${SITE_PACKAGES_DIR}"

# --- 2. Check for write permissions ---
if [ ! -w "$SITE_PACKAGES_DIR" ]; then
    echo -e "${RED}Error: No write permission for ${SITE_PACKAGES_DIR}.${NC}"
    echo "Please run this script with sufficient permissions (e.g., using 'sudo ./uninstall_patch.sh')."
    exit 1
fi

# --- 3. Remove the files ---
echo "Removing patch files..."
# Use "rm -f" to avoid errors if a file is already missing
rm -vf "${SITE_PACKAGES_DIR}/${PATCH_CORE_FILE}"
rm -vf "${SITE_PACKAGES_DIR}/${PATCH_LOADER_FILE}"
rm -vf "${SITE_PACKAGES_DIR}/${PTH_FILE}"

echo -e "\n${GREEN}SGLang patch uninstalled successfully!${NC}"
