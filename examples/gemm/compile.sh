#!/usr/bin/env bash
CACHE_HASH=16b23bf81401b553594da4db04520d6ebc6100d7dd00ba553c9d2e9af22c7797
CACHE_DIR=/root/.tilelang/cache/${CACHE_HASH}

hipcc -std=c++17 -fPIC -Rpass-analysis=kernel-resource-usage --save-temps -g --offload-arch=gfx950 --shared "${CACHE_DIR}/host_kernel.cu" -I/A/tilelang/3rdparty/composable_kernel/include -I/A/tilelang/3rdparty/../src -o "${CACHE_DIR}/kernel_lib.so"
