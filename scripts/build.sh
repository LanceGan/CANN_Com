#!/bin/bash
# f:/Projects/CANN_Com/scripts/build.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# Auto-detect generator
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "mingw"* || "$OSTYPE" == "cygwin" ]]; then
    GENERATOR="MinGW Makefiles"
else
    GENERATOR="Unix Makefiles"
fi

echo "=== Building CANN_Com ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -G "$GENERATOR"
cmake --build . -j$(nproc 2>/dev/null || echo 4)
echo "=== Build complete ==="

echo "=== Running tests ==="
ctest --output-on-failure
echo "=== Tests complete ==="
