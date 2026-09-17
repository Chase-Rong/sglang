#!/usr/bin/env bash
set -eo pipefail

fsa_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fsa_work=${SGLANG_FSA_WORK_DIR:?Set the assigned isolated work directory}
fsa_project=${1:?Supply a new project directory under SGLANG_FSA_WORK_DIR}
fsa_cann=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}
case "$(realpath -m "$fsa_project")" in
    "$(realpath -m "$fsa_work")"/*) ;;
    *) echo "Build project must be inside SGLANG_FSA_WORK_DIR" >&2; exit 1 ;;
esac
source "$fsa_cann/set_env.sh"
export TMPDIR="$fsa_work/tmp"
mkdir -p "$TMPDIR"
python3 "$fsa_repo/scripts/npu/prepare_fsa_opp.py" \
    --cann "$fsa_cann" --output "$fsa_project"
cmake -S "$fsa_project" -B "$fsa_project/build" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_COMPUTE_UNIT=ascend950 \
    -DASCEND_CANN_PACKAGE_PATH="$fsa_cann" \
    -DENABLE_BINARY_PACKAGE=ON -DENABLE_SOURCE_PACKAGE=ON -DENABLE_TEST=OFF \
    -Dvendor_name=sgl_fsa_a5 -DCMAKE_INSTALL_PREFIX="$fsa_project/install"
cmake --build "$fsa_project/build" -j"${MAX_JOBS:-2}"
cmake --build "$fsa_project/build" --target binary -j"${MAX_JOBS:-2}"
cmake --install "$fsa_project/build"
