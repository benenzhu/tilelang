当前生成的gemm 根据每个warp 处理的是一个64 * 64的块，那么划分到256*256的C的block就是：
0 0 4 4
1 1 5 5
2 2 6 6
3 3 7 7

我想改成让他生成 32 * 64的块，然后划分到256*256的C block是： 
0 4 0 4
1 5 1 5
2 6 2 6
3 7 3 7
0 4 0 4
1 5 1 5
2 6 2 6
3 7 3 7
有什么办法么？

for k in range(127):
    load(As[k][0])     // 4 load_128
    load(Bs[k][0])     // 8 load_128
    GL(As[k + 1][1])    // 2 buffer
    mma
    --------------------
    load(Bs[k][1])     // 8 load_128
    GL(As[k + 2][0])   // 2 buffer
    mma
    --------------------
    load(As[k][1])      // 4 load_128
    Gl(Bs[k + 2][0])    // 2 buffer
    mma
    --------------------
    GL(Bs[k + 2][1])    // 2 buffer
    mma