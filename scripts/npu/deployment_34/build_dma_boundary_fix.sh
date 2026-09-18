#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
RELEASE=/home/j00628475/sglang_release_20260917_f524c79a
SRC=$RELEASE/sglang_remote/native/fsa
BUILD=$RELEASE/fsa-binding-dma-segments
mkdir -p "$BUILD"
c++ -std=c++17 -Wall -Wextra -Werror -I "$SRC" "$SRC/tests/dma_segments_test.cpp" -o "$BUILD/dma_segments_test"
"$BUILD/dma_segments_test"
cmake -S "$SRC" -B "$BUILD" \
    -DFSA_SOURCE_ROOT=/home/j00628475/request_budget_copy64_20260916_v2/kernel_source \
    -DSOC_VERSION=Ascend950PR_9599
cmake --build "$BUILD" -j 8
sha256sum "$BUILD/libsgl_fsa_extension.so"
