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

# Use system-installed GTest (MSYS2 UCRT64 on Windows)
CMAKE_EXTRA_ARGS=""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "mingw"* || "$OSTYPE" == "cygwin" ]]; then
    CMAKE_EXTRA_ARGS="-DCMAKE_PREFIX_PATH=/f/msys64/ucrt64"
fi

cmake .. -G "$GENERATOR" $CMAKE_EXTRA_ARGS
cmake --build . -j$(nproc 2>/dev/null || echo 4)
echo "=== Build complete ==="

echo "=== Running tests ==="
# Run each test binary directly to avoid MinGW/GTest teardown segfault
# (known issue: GTest global cleanup segfaults on MinGW after all tests pass)
TESTS_PASSED=0
TESTS_FAILED=0
for test_bin in tests/test_*.exe; do
    name=$(basename "$test_bin" .exe)
    output=$("$test_bin" 2>&1) || true
    if echo "$output" | grep -q "\[  PASSED  ]"; then
        count=$(echo "$output" | grep -c "\[       OK \]")
        echo "  $name: PASSED ($count tests)"
        TESTS_PASSED=$((TESTS_PASSED + count))
    else
        echo "  $name: FAILED"
        echo "$output" | grep "\[  FAILED  \]" || true
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
done
echo "=== $TESTS_PASSED tests passed across 13 suites ==="
if [ $TESTS_FAILED -gt 0 ]; then
    echo "ERROR: $TESTS_FAILED test suite(s) failed"
    exit 1
fi
echo "=== Tests complete ==="
