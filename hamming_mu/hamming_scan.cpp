// hamming_scan.cpp — host 驱动:MX1P 上的 Hamming 扫描与 top-k。
//
// Build: 见 build.sh(需 XCENA SDK v1.4.9、PXL、MU LLVM 工具链)

#include <cstdio>
#include "pxl/pxl.hpp"

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;
    auto context = pxl::createContext(0);
    if (!context)
    {
        printf("createContext failed\n");
        return 1;
    }
    printf("skeleton: context created\n");
    pxl::destroyContext(context);
    return 0;
}
