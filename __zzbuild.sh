mkdir build 
cp 3rdparty/tvm/cmake/config.cmake build
cd build
# echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUDA ON)" >> config.cmake 
echo "set(USE_RELAY_DEBUG ON)" >> config.cmake
echo "set(CMAKE_BUILD_TYPE Debug)" >> config.cmake
# or echo "set(USE_ROCM ON)" >> config.cmake to enable ROCm runtime
cmake -DCMAKE_BUILD_TYPE=Debug ..
mold -run make -j46

exit 0

#%% cov



rm -rf result __1.start.info __2.start.info
find build/  -type f -name "*.gcda" -exec rm -f {} \;
# build/bin/clangd -lit-test < clang-tools-extra/clangd/test/start.py
fastcov --search-directory=build/ --lcov -o __1.start.info
genhtml -j 100 -s --sort --no-function-coverage --show-navigation --ignore-errors source --ignore-errors negative -o result report.info

# find build/  -type f -name "*.gcda" -exec rm -f {} \;
# build/bin/clangd -lit-test < clang-tools-extra/clangd/test/ast.py
# fastcov --search-directory=build/tools/clang/tools/extra/clangd/ --lcov -o __2.start.info
# genhtml -j 100 -s --sort --no-function-coverage --show-navigation --ignore-errors source --ignore-errors negative -o result __2.start.info -b __1.start.info