#!/bin/bash
# Script to prepare the build environment for pytorch3d.
#
# Example usage:
#   ./prepare_for_build.sh fc6a6b895185a502c4064fc5ea7ecb760fd1a0d6

set -euxo pipefail

# When run from CI, this script is in build_scripts/prepare_for_build.sh
# and needs to reference patches from that directory
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pytorch3d-version>"
    echo "Example: $0 fc6a6b895185a502c4064fc5ea7ecb760fd1a0d6"
    exit 1
fi

PYTORCH3D_VERSION=$1

# Ensure that the pytorch3d version is supported.
if [ ! -d "${SCRIPT_DIR}/patches/${PYTORCH3D_VERSION}" ]; then
    echo "Error: patches/${PYTORCH3D_VERSION} directory does not exist"
    exit 1
fi

# Apply patches.
for patch in "${SCRIPT_DIR}/patches/${PYTORCH3D_VERSION}"/*.patch; do
    patch -p1 -d ${ROOT} -i ${patch}
done
