cmake_minimum_required(VERSION 3.11)
project(rabitq_scan)

# 结构照抄 hamming_mu/CMakeLists.txt:host 程序链接 PXL,MU kernel 用
# ExternalProject 以 MU LLVM 工具链单独编译成 mu_kernel/mu_kernel.mubin。

set(shared_libs)
if (NOT XCENA_SDK_CMAKE_BUILD)
    find_package(pxl REQUIRED)
    message(STATUS "PXL library found at: ${pxl_DIR}")
    LIST(APPEND shared_libs pxl::pxl)
else()
    LIST(APPEND shared_libs pxl)
endif()

add_executable(rabitq_scan rabitq_scan.cpp)
target_compile_features(rabitq_scan PRIVATE cxx_std_17)
target_include_directories(rabitq_scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rabitq_scan ${shared_libs})
install(TARGETS rabitq_scan RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

# bench/:Preload Region 容量探测(只链接 PXL,没有 kernel)
add_executable(preload_probe bench/preload_probe.cpp)
target_compile_features(preload_probe PRIVATE cxx_std_17)
target_link_libraries(preload_probe ${shared_libs})
install(TARGETS preload_probe RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bench)

include(ExternalProject)
ExternalProject_Add(build_mu_${PROJECT_NAME}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mu_kernel
    BUILD_ALWAYS TRUE
    CMAKE_ARGS
        -DMU_LIB_PATH=$ENV{MU_LIB_PATH}
        -DMU_LLVM_PATH=$ENV{MU_LLVM_PATH}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)
