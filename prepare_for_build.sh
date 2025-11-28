#!/bin/bash
# Script to prepare the build environment for PyTorch3D.
#
# Example usage:
#   ./prepare_for_build.sh v0.7.9

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pytorch3d_version>"
    echo "Example: $0 v0.7.9"
    exit 1
fi

PYTORCH3D_VERSION=$1

# Apply patches.
patch_dir="${ROOT}/build_scripts/patches/${PYTORCH3D_VERSION}"

# Not all PyTorch3D versions need patches.
if [ ! -d "${patch_dir}" ]; then
    echo "Warning: nothing to patch: patches/${PYTORCH3D_VERSION} directory does not exist"
else
    for patch in "${patch_dir}"/*.patch; do
        # Skip if no patch files exist (only .gitkeep)
        if [ -f "${patch}" ]; then
            patch -p1 -d "${ROOT}" -i "${patch}"
        fi
    done
fi
