cmake_minimum_required(VERSION 3.20)
project(sgl_lim_copy_sfa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION "" CACHE STRING "Exact target SoC")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN root")
set(LIM_REFERENCE_ROOT "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/lightning_indexer"
    CACHE PATH "Pinned CANN A5 Lightning Indexer source")
if(NOT SOC_VERSION MATCHES "^Ascend950")
    message(FATAL_ERROR "Set SOC_VERSION to the exact Ascend950 target")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_reference.py"
    "${LIM_REFERENCE_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/reference"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib,torch,torch_npu; print(pathlib.Path(torch.__file__).parent); print(pathlib.Path(torch_npu.__file__).parent); print(int(torch.compiled_with_cxx11_abi()))"
    OUTPUT_VARIABLE LIM_PATHS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" LIM_PATHS "${LIM_PATHS}")
list(GET LIM_PATHS 0 LIM_TORCH)
list(GET LIM_PATHS 1 LIM_TORCH_NPU)
list(GET LIM_PATHS 2 LIM_ABI)
list(APPEND CMAKE_PREFIX_PATH "${LIM_TORCH}/share/cmake")
find_package(Torch REQUIRED)
file(GLOB LIM_CMAKE
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
if(NOT LIM_CMAKE)
    message(FATAL_ERROR "CANN AscendC CMake module not found")
endif()
list(GET LIM_CMAKE 0 LIM_MODULE)
include("${LIM_MODULE}")
ascendc_library(sgl_lim_indexer_kernel STATIC op_kernel/lim_indexer.cpp op_kernel/lim_manage.cpp op_kernel/lim_indexer_manage.cpp)
ascendc_include_directories(sgl_lim_indexer_kernel PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/reference")
ascendc_library(sgl_lim_copy_kernel STATIC op_kernel/lim_copy_sfa.cpp)
ascendc_compile_options(sgl_lim_copy_kernel PRIVATE --cce-auto-sync=off
    "SHELL:-mllvm -cce-vf-remove-membar=false" "SHELL:-mllvm -cce-aicore-hoist-movemask=false")
add_library(sgl_lim_copy_sfa SHARED registration.cpp manage_registration.cpp fused_registration.cpp copy_registration.cpp)
set(LIM_CONFIGURED_DIR "${CMAKE_CURRENT_BINARY_DIR}/configured")
add_custom_command(
    OUTPUT "${LIM_CONFIGURED_DIR}/configured_indexer.cpp" "${LIM_CONFIGURED_DIR}/configured_indexer.h"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_configured_launch.py"
        "${CMAKE_CURRENT_BINARY_DIR}" "${LIM_CONFIGURED_DIR}"
    DEPENDS sgl_lim_indexer_kernel "${CMAKE_CURRENT_SOURCE_DIR}/prepare_configured_launch.py"
    VERBATIM)
target_sources(sgl_lim_copy_sfa PRIVATE "${LIM_CONFIGURED_DIR}/configured_indexer.cpp")
target_compile_definitions(sgl_lim_copy_sfa PRIVATE _GLIBCXX_USE_CXX11_ABI=${LIM_ABI})
target_include_directories(sgl_lim_copy_sfa PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${LIM_CONFIGURED_DIR}"
    "${LIM_TORCH_NPU}/include" "${LIM_TORCH_NPU}/include/third_party/acl/inc"
    "${ASCEND_CANN_PACKAGE_PATH}/include" ${Python3_INCLUDE_DIRS})
target_link_directories(sgl_lim_copy_sfa PRIVATE "${LIM_TORCH_NPU}/lib" "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(sgl_lim_copy_sfa PRIVATE sgl_lim_indexer_kernel sgl_lim_copy_kernel ${TORCH_LIBRARIES} torch_npu ascendcl)
