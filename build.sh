#!/bin/bash
# TileLang ROCm Build Script
#
# Prerequisites:
#   - ROCm SDK installed (default at /opt/rocm)
#   - Python 3.9+ with pip
#   - Required system packages:
#     sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev \
#         zlib1g-dev build-essential cmake libedit-dev libxml2-dev
#
# NOTE: If you encounter GTest cmake errors like:
#   "Neither GTest::GTest nor GTest::gtest targets defined IMPORTED_LOCATION"
# You need to remove libgtest-dev:
#   sudo apt-get remove -y libgtest-dev
sudo apt-get remove -y libgtest-dev
set -e
set -x

# ==============================================================================
# Configuration
# ==============================================================================

# Set ROCm SDK path (default: /opt/rocm)
# Change this if your ROCm is installed elsewhere
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# Build options
EDITABLE_MODE="${EDITABLE_MODE:-1}"  # Set to 0 for non-editable install
VERBOSE="${VERBOSE:-1}"
NO_BUILD_ISOLATION="${NO_BUILD_ISOLATION:-1}"  # Recommended for faster rebuilds

# ==============================================================================
# Build
# ==============================================================================

echo "=========================================="
echo "TileLang ROCm Build"
echo "=========================================="
echo "ROCm Path: $ROCM_PATH"
echo "Editable Mode: $EDITABLE_MODE"
echo ""

# Check ROCm installation
if [ ! -d "$ROCM_PATH" ]; then
    echo "Error: ROCm SDK not found at $ROCM_PATH"
    echo "Please install ROCm or set ROCM_PATH environment variable"
    exit 1
fi

# Install dev dependencies if using no-build-isolation
if [ "$NO_BUILD_ISOLATION" = "1" ]; then
    echo "Installing dev dependencies..."
    pip install -r requirements-dev.txt
fi

# Build command
BUILD_CMD="pip install"

if [ "$EDITABLE_MODE" = "1" ]; then
    BUILD_CMD="$BUILD_CMD -e "
fi

BUILD_CMD="$BUILD_CMD ."

if [ "$VERBOSE" = "1" ]; then
    BUILD_CMD="$BUILD_CMD -v"
fi

if [ "$NO_BUILD_ISOLATION" = "1" ]; then
    BUILD_CMD="$BUILD_CMD --no-build-isolation --break-system-packages "
fi

# Set USE_ROCM environment variable
# If ROCm is at /opt/rocm, USE_ROCM=1 is sufficient
# Otherwise, set it to the ROCm SDK path
if [ "$ROCM_PATH" = "/opt/rocm" ]; then
    export USE_ROCM=1
else
    export USE_ROCM="$ROCM_PATH"
fi

echo "Running: USE_ROCM=$USE_ROCM $BUILD_CMD"
echo ""

eval $BUILD_CMD

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Verify installation:"
echo "  python -c \"import tilelang; print(tilelang.__version__)\""
