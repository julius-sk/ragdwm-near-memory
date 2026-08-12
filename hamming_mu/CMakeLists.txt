cmake_minimum_required(VERSION 3.11)
project(hamming_scan)

set(src
    hamming_scan.cpp
)

set(include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(shared_libs)

if (NOT XCENA_SDK_CMAKE_BUILD)
    find_package(pxl REQUIRED)
    message(STATUS "PXL library found at: ${pxl_DIR}")
    LIST(APPEND shared_libs pxl::pxl)
else()
    LIST(APPEND shared_libs pxl)
endif()

foreach(file ${src})
    get_filename_component(target ${file} NAME_WE)
    add_executable(${target} ${file})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ${include})
    target_link_libraries(${target} ${shared_libs})
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# --- A 组主机基线(C + OpenMP)------------------------------------------
# 与 PXL、与设备完全无关,只是个多线程 CPU 实现。
# **刻意做成可选**:OpenMP 找不到就跳过并给出手工编译命令,绝不让一个基线
# 工具拖垮已经跑通的设备侧构建。
find_package(OpenMP COMPONENTS C QUIET)
if (OpenMP_C_FOUND)
    add_executable(bench_host_c bench_host_c.c)
    target_link_libraries(bench_host_c PRIVATE OpenMP::OpenMP_C)
    if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench_host_c PRIVATE -O3 -march=native)
    endif()
    install(TARGETS bench_host_c RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "bench_host_c: OpenMP found -> will build")
else()
    message(STATUS "bench_host_c: OpenMP NOT found -> skipped. 手工编译:")
    message(STATUS "  gcc -O3 -march=native -fopenmp -o bench_host_c bench_host_c.c")
endif()

include(ExternalProject)
ExternalProject_Add(build_mu_${PROJECT_NAME}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mu_kernel
    BUILD_ALWAYS TRUE
    CMAKE_ARGS
        -DMU_LIB_PATH=$ENV{MU_LIB_PATH}
        -DMU_LLVM_PATH=$ENV{MU_LLVM_PATH}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)
