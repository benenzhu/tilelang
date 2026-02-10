#!/usr/bin/env bash
CACHE_HASH=161e7ab5309abbc4eded047e896e9f14846dd9bfaa8075ce7e715cea648d6712
CACHE_DIR=/root/.tilelang/cache/${CACHE_HASH}

hipcc -std=c++17 -fPIC -Rpass-analysis=kernel-resource-usage --save-temps -g --offload-arch=gfx950 --shared "${CACHE_DIR}/host_kernel.cu" -I/A/tilelang/3rdparty/composable_kernel/include -I/A/tilelang/3rdparty/../src -o "${CACHE_DIR}/kernel_lib.so"
