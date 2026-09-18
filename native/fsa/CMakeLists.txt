cmake_minimum_required(VERSION 3.20)
project(sgl_fsa_extension LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "Exact SoC reported by the target CANN runtime")
set(FSA_SOURCE_ROOT "" CACHE PATH "Exported sgl-kernel-npu sources containing the FSA wrapper")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN root")
set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver" CACHE PATH "Ascend driver headers and libraries")
if(NOT EXISTS "${FSA_SOURCE_ROOT}/csrc/fused_sparse_attention_overlap/fsa_lru_planner.cpp")
    message(FATAL_ERROR "FSA_SOURCE_ROOT must point to the exported FSA source bundle")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib,torch,torch_npu; print(pathlib.Path(torch.__file__).parent); print(pathlib.Path(torch_npu.__file__).parent); print(int(torch.compiled_with_cxx11_abi()))"
    OUTPUT_VARIABLE FSA_PYTHON_PATHS OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" FSA_PYTHON_PATHS "${FSA_PYTHON_PATHS}")
list(GET FSA_PYTHON_PATHS 0 FSA_TORCH_ROOT)
list(GET FSA_PYTHON_PATHS 1 FSA_TORCH_NPU_ROOT)
list(GET FSA_PYTHON_PATHS 2 FSA_CXX11_ABI)
list(APPEND CMAKE_PREFIX_PATH "${FSA_TORCH_ROOT}/share/cmake")
find_package(Torch REQUIRED)
file(GLOB FSA_ASCENDC_MODULES
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
if(NOT FSA_ASCENDC_MODULES)
    message(FATAL_ERROR "CANN AscendC CMake module was not found")
endif()
list(GET FSA_ASCENDC_MODULES 0 FSA_ASCENDC_MODULE)
include("${FSA_ASCENDC_MODULE}")
set(FSA_CSRC "${FSA_SOURCE_ROOT}/csrc")
ascendc_library(fsa_unidex_kernel STATIC
    "${FSA_CSRC}/sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp")
ascendc_include_directories(fsa_unidex_kernel PRIVATE "${FSA_CSRC}/utils/kernel")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.h" "#pragma once\n")
add_library(sgl_fsa_extension SHARED registration.cpp
    external_host_mapping.cpp
    prefill_dma.cpp
    fused_sparse_attention_overlap.cpp
    "${FSA_CSRC}/fused_sparse_attention_overlap/fsa_lru_planner.cpp"
    "${FSA_CSRC}/sparsity_driven_kv_offload/shm_allocator/shm_allocator.cpp"
    "${FSA_CSRC}/sparsity_driven_kv_offload/shm_allocator/shm_allocator_op.cpp"
    "${FSA_CSRC}/sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp")
target_compile_definitions(sgl_fsa_extension PRIVATE _GLIBCXX_USE_CXX11_ABI=${FSA_CXX11_ABI})
target_include_directories(sgl_fsa_extension PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}" "${FSA_CSRC}" "${FSA_CSRC}/utils"
    "${FSA_CSRC}/fused_sparse_attention_overlap"
    "${FSA_TORCH_NPU_ROOT}/include" "${FSA_TORCH_NPU_ROOT}/include/third_party/acl/inc"
    "${ASCEND_CANN_PACKAGE_PATH}/include" "${ASCEND_DRIVER_PATH}/include"
    ${Python3_INCLUDE_DIRS})
target_link_directories(sgl_fsa_extension PRIVATE
    "${FSA_TORCH_NPU_ROOT}/lib" "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${ASCEND_DRIVER_PATH}/lib64/driver")
target_link_libraries(sgl_fsa_extension PRIVATE fsa_unidex_kernel ${TORCH_LIBRARIES}
    torch_npu ascendcl ascend_hal Python3::Python dl)
