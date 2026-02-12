#!/usr/bin/env bash
CACHE_HASH=dca8f57ec544be2740fcfbf506299b7aa933f9bf79c2f529b4e1e37c797a0757
CACHE_DIR=/root/.tilelang/cache/${CACHE_HASH}

hipcc -std=c++17 -fPIC -Rpass-analysis=kernel-resource-usage --save-temps -g --offload-arch=gfx950 --shared "${CACHE_DIR}/host_kernel.cu" -I/A/tilelang/3rdparty/composable_kernel/include -I/A/tilelang/3rdparty/../src -o "${CACHE_DIR}/kernel_lib.so"
