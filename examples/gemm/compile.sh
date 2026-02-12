#!/usr/bin/env bash
CACHE_HASH=df5bb017d458a477c1bd3030b83831ec20c6a2ece1f22c6f32633c54525ef6bf
CACHE_DIR=/root/.tilelang/cache/${CACHE_HASH}

hipcc -std=c++17 -fPIC -Rpass-analysis=kernel-resource-usage --save-temps -g --offload-arch=gfx950 --shared "${CACHE_DIR}/host_kernel.cu" -I/A/tilelang/3rdparty/composable_kernel/include -I/A/tilelang/3rdparty/../src -o "${CACHE_DIR}/kernel_lib.so"
