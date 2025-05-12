.PHONY: all clean passes

all:
	cd build && mold -run make -j48 
	cd ..
	TILELANG_CLEAR_CACHE=1 python examples/gemm/example_gemm.py


clearcov:
	rm -rf result __1.start.info __2.start.info
	find build/  -type f -name "*.gcda" -exec rm -f {} \;	
gencov:
	fastcov --search-directory=build/ --lcov -o __1.start.info
	genhtml -j 100 -s --sort --no-function-coverage --show-navigation --ignore-errors source --ignore-errors negative -o result __1.start.info
p: # passes
	python __gen_pass.py



clean:
	rm -rf build/*

# Default target runs build and example
default: all
