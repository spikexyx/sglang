#!/bin/bash

# install_patch.sh: A script to install the SGLang non-intrusive patch.

# --- Configuration ---
# Names of the files to be installed.
PATCH_CORE_FILE="sglang_weight_hook_patch_core.py"
PATCH_LOADER_FILE="sglang_patch_loader.py"
PTH_FILE="sglang_injector.pth"

# --- Style Definitions ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Exit immediately if a command exits with a non-zero status.
set -e

echo -e "${YELLOW}Starting SGLang Patch Installation...${NC}"

# --- 1. Find the script's own directory to locate source files ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
echo "Searching for source files in: ${SCRIPT_DIR}"

# --- 2. Check if source files exist ---
if [ ! -f "${SCRIPT_DIR}/${PATCH_CORE_FILE}" ] || \
   [ ! -f "${SCRIPT_DIR}/${PATCH_LOADER_FILE}" ] || \
   [ ! -f "${SCRIPT_DIR}/${PTH_FILE}" ]; then
    echo -e "${RED}Error: One or more source files not found in the script's directory.${NC}"
    echo "Please ensure ${PATCH_CORE_FILE}, ${PATCH_LOADER_FILE}, and ${PTH_FILE} are in the same directory as this script."
    exit 1
fi
echo -e "${GREEN}Source files found.${NC}"

# --- 3. Find the active Python environment's site-packages directory ---
echo "Detecting active Python environment..."
# Prefer 'python3' if available, otherwise fall back to 'python'
PYTHON_EXEC=$(command -v python3 || command -v python)

if [ -z "$PYTHON_EXEC" ]; then
    echo -e "${RED}Error: Could not find 'python' or 'python3' in your PATH.${NC}"
    echo "Please activate your target Python environment first."
    exit 1
fi
echo "Using Python executable: ${PYTHON_EXEC}"

# Get the first site-packages path. The python command is robust against empty results.
SITE_PACKAGES_DIR=$($PYTHON_EXEC -c "import site; print(site.getsitepackages()[0] if site.getsitepackages() else '')")

if [ -z "$SITE_PACKAGES_DIR" ] || [ ! -d "$SITE_PACKAGES_DIR" ]; then
    echo -e "${RED}Error: Could not determine a valid site-packages directory.${NC}"
    echo "Is your Python environment correctly configured?"
    exit 1
fi
echo "Target site-packages directory: ${SITE_PACKAGES_DIR}"

# --- 4. Check for write permissions ---
if [ ! -w "$SITE_PACKAGES_DIR" ]; then
    echo -e "${RED}Error: No write permission for ${SITE_PACKAGES_DIR}.${NC}"
    echo "Please run this script with sufficient permissions (e.g., using 'sudo ./install_patch.sh' if installing to a system-wide python)."
    exit 1
fi

# --- 5. Copy the files ---
echo "Copying patch files..."
cp -v "${SCRIPT_DIR}/${PATCH_CORE_FILE}" "${SITE_PACKAGES_DIR}/"
cp -v "${SCRIPT_DIR}/${PATCH_LOADER_FILE}" "${SITE_PACKAGES_DIR}/"
cp -v "${SCRIPT_DIR}/${PTH_FILE}" "${SITE_PACKAGES_DIR}/"

echo -e "\n${GREEN}SGLang patch installed successfully!${NC}"
echo "The patch is now active for the Python environment at ${PYTHON_EXEC}."
echo "You can now run 'python -m sglang.launch_server ...' directly."
