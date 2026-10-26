#!/usr/bin/env bash
# build_bench.sh —— 构建 bench/ 里的程序。
#   rabitq_scan + MU kernel + preload_probe:走 rabitq_mu/build.sh(CMake + MU LLVM)
#   host_rabitq、stream_triad:纯主机 C,gcc 直接编
set -e
cd "$(dirname "$0")"
(cd .. && bash build.sh)
gcc -O3 -march=native -fopenmp -o host_rabitq host_rabitq.c -lm
gcc -O3 -march=native -fopenmp -o stream_triad stream_triad.c
ls -l ../rabitq_scan ../mu_kernel/mu_kernel.mubin preload_probe host_rabitq stream_triad
