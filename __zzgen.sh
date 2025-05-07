rm -rf result __1.start.info __2.start.info
find build/  -type f -name "*.gcda" -exec rm -f {} \;
# your command here...
fastcov --search-directory=build/ --lcov -o __1.start.info
genhtml -j 100 -s --sort --no-function-coverage --show-navigation --ignore-errors source --ignore-errors negative -o result __1.start.info