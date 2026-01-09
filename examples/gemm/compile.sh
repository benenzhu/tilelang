set -e
set -x
hipcc -std=c++17 -fPIC --offload-arch=gfx942 --shared -I/A/tilelang-ali/3rdparty/composable_kernel/include -I/A/tilelang-ali/3rdparty/../src  /root/.tilelang/cache/$1/host_kernel.cu -o /root/.tilelang/cache/$1/kernel_lib.so
