#!/usr/bin/env bash
CACHE_HASH=683ac3259a15d32d47fa729e636e891ddd7671942bddf3dc5cb505c501f09975
CACHE_DIR=/root/.tilelang/cache/${CACHE_HASH}

hipcc -std=c++17 -fPIC -Rpass-analysis=kernel-resource-usage --save-temps -g --offload-arch=gfx950 --shared "${CACHE_DIR}/host_kernel.cu" -I/root/tilelang/3rdparty/composable_kernel/include -I/root/tilelang/3rdparty/../src -o "${CACHE_DIR}/kernel_lib.so"
