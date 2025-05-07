rm -rf result __1.start.info __2.start.info
find build/  -type f -name "*.gcda" -exec rm -f {} \;
# your command here...
TILELANG_CLEAR_CACHE=1 nohup python examples/gemm/example_gemm.py >nohup.py 2>&1  &
fastcov --search-directory=build/ --lcov -o __1.start.info
genhtml -j 100 -s --sort --no-function-coverage --show-navigation --ignore-errors source --ignore-errors negative -o result __1.start.info