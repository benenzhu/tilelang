mkidr build 
cp 3rdparty/tvm/cmake/config.cmake build
cd build
# echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUDA ON)" >> config.cmake 
# or echo "set(USE_ROCM ON)" >> config.cmake to enable ROCm runtime
cmake ..
mold -run make -j