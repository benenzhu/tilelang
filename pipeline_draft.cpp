[128][64] As[2][2]
[128][64] Bs[2][2]

[64][64] A_tile
[32][64] B_tile

row__0_1
col__0_3

Gl(Bs[0][0],   b[col*2  ][0])
Gl(As[0][0],   a[row*2  ][0])
Gl(Bs[0][1],   b[col*2+1][0])
Gl(As[0][1],   a[row*2+1][0])      128 * 64 element => /512 * 8 = 2 instr
                                   how to load for each warp?

Gl(Bs[1][0],   b[col*2  ][1])
Gl(As[1][0],   a[row*2  ][1])
Gl(Bs[1][1],   b[col*2+1][1])

vmcnt(6) :// 上面的 As[0][1] 都是能用的了.. 先不考虑 pingpong 这里..
vmcnt(0) :
for:
-----------------------------
vmcnt(12)
load(B_tile_0, Bs[0][0](w_col)) // 7 * 2 = 14
vmcnt(10)
load(A_tile,   As[0][0](w_row)) // 5 * 2 = 10
Gl(As[1][1],   a[row*2+1][1])
mma(A_tile, B_tile_0)
------------------------------
vmcnt(10)
load(B_tile_1, Bs[0][1](col)) // 5 * 2 = 10
Gl(Bs[0][0],   b[col*2  ][0])
mma(A_tile, B_tile_1)
------------------------------
vmcnt(10)
load(A_tile, As[0][1](row)) // 5 * 2 = 10
lgkmcnt(0)
Gl(As[0][0],   a[row*2  ][0])
mma(A_tile, B_tile_0)
----------------------------
vmcnt(10)
load(B_tile_0, Bs[1][0])  // 5 * 2 = 10
Gl(Bs[0][1],   b[col*2+1][0])
mma(A_tile, B_tile_1)
----------------------------
vmcnt(10)
load(A_tile,   As[1][0][w_row]) // 5 * 2 = 10
Gl(As[0][1],  ....)
mma(A_tile, B_tile_0)
----------------------------
vmcnt(10)
load(B_tile_1, Bs[1][1][w_col]) // 5 * 2 = 10
Gl(Bs[1][0],    b[col*2])
lkgmcnt(0)
mma(A_tile, B_tile_1)
----------------------------
vmcnt(10)
load(A_tile, As[1][1](warp_row)) // 5 * 2 = 10
Gl(As[1][0], a[row*2][1])
mma(A_tile, B_tile_0)
----------------------------
Gl(Bs[1][1], b[col*2+1][1])
mma(A_tile, B_tile_1)



for:
-----------------------------
vmcnt(xx)
load(B_tile_0, Bs[k&1][0](w_col)) // 7 * 2 = 14
vmcnt(xx)
load(A_tile,   As[k&1][0](w_row)) // 5 * 2 = 10
Gl(As[(k+1)&1][1],   a[row*2+1][1])
mma(A_tile, B_tile_0)
------------------------------
vmcnt(xx)
load(B_tile_1, Bs[k&1][1](col)) // 5 * 2 = 10
Gl(Bs[k&1][0],   b[col*2  ][0])
mma(A_tile, B_tile_1)
------------------------------
vmcnt(xx)
load(A_tile, As[k&1][1](row)) // 5 * 2 = 10
lgkmcnt(0)
Gl(As[k&1][0],   a[row*2  ][0])
mma(A_tile, B_tile_0)
----------------------------
vmcnt(10)
load(B_tile_0, Bs[k&1][0])  // 5 * 2 = 10
Gl(Bs[k&1][1],   b[col*2+1][0])
mma(A_tile, B_tile_1)
// ----------------------------
// vmcnt(10)
// load(A_tile,   As[1][0][w_row]) // 5 * 2 = 10
// Gl(As[0][1],  ....)
// mma(A_tile, B_tile_0)
// ----------------------------
// vmcnt(10)
// load(B_tile_1, Bs[1][1][w_col]) // 5 * 2 = 10
// Gl(Bs[1][0],    b[col*2])
// lkgmcnt(0)
// mma(A_tile, B_tile_1)
// ----------------------------
// vmcnt(10)
// load(A_tile, As[1][1](warp_row)) // 5 * 2 = 10
// Gl(As[1][0], a[row*2][1])
// mma(A_tile, B_tile_0)
// ----------------------------
// Gl(Bs[1][1], b[col*2+1][1])
// mma(A_tile, B_tile_1)
//


    
// before loop
GL(As[0][0])
GL(Bs[0][0])
GL(As[0][1])
GL(Bs[0][1])
GL(As[1][0])
GL(Bs[1][0])
GL(Bs[1][1])

// k = 0
GL(As[1][1])
load(As[0][0])   
load(Bs[0][0])   mma
--------------------
GL(As[0][0])
load(Bs[0][1])   mma
--------------------
Gl(Bs[0][0])
load(As[0][1])   mma
--------------------
GL(Bs[0][1])     mma
--------------------
// k = 1
GL(As[0][1])
load(As[1][0])   
load(Bs[1][0])   mma
--------------------
GL(As[1][0])
load(Bs[1][1])   mma
--------------------
Gl(Bs[1][0])
load(As[1][1])   mma
--------------------
GL(Bs[1][1])     mma
--------------------
// k = 2
GL(As[1][1])
load(As[0][0])   
load(Bs[0][0])   mma 
--------------------
GL(As[0][0])
load(Bs[0][1])   mma
--------------------
Gl(Bs[0][0])
load(As[0][1])   mma
--------------------
GL(Bs[0][1])     mma⏎                                    





一个tile: 


G::load(A) :: // (256 * 64 / 512) = 32 ele.  32 / 8 = ... 4 load


local: 

A[0] B[0] B[1]:   A: 64 * 64 / 64 = 64 ele. 64 / 8 = 8 load...  * 3 = 24 load.





// k = 0
load(As[0][0])  // 8 load_128
load(Bs[0][0])  // 4 load_128
GL(As[1][1])    // 2 buffer
mma
--------------------
load(Bs[0][1])  // 4 load_128
GL(As[0][0])    // 2 buffer
mma
--------------------
load(As[0][1])  // 8 load_128
Gl(Bs[0][0])    // 2 buffer
mma
--------------------
GL(Bs[0][1])    // 2 buffer
mma
--------------------
// k = 1
GL(As[0][1])
load(As[1][0])   
load(Bs[1][0])   mma
--------------------
GL(As[1][0])
load(Bs[1][1])   mma
--------------------
Gl(Bs[1][0])
load(As[1][1])   mma
--------------------
GL(Bs[1][1])     mma
--------------------
// k = 2
GL(As[1][1])
load(As[0][0])   
load(Bs[0][0])   mma 
--------------------
GL(As[0][0])
load(Bs[0][1])   mma
--------------------
Gl(Bs[0][0])
load(As[0][1])   mma
--------------------
GL(Bs[0][1])     mma⏎                                    


// k = 0
load(As[k][0])  // 8 load_128
load(Bs[k][0])  // 4 load_128
GL(As[k + 1][1])    // 2 buffer
mma
--------------------
load(Bs[k][1])  // 4 load_128
GL(As[k + 2][0])    // 2 buffer
mma
--------------------
load(As[k][1])  // 8 load_128
Gl(Bs[k + 2][0])    // 2 buffer
mma
--------------------
GL(Bs[k + 2][1])    // 2 buffer
mma