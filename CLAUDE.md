# TileLang Development Notes

## Build

```bash
bash ../backup/build.sh          # incremental (fast, cmake only)
bash ../backup/build.sh --full   # full pip install
```

Test: `python examples/gemm/example_gemm.py`

## Architecture: GEMM Kernel on AMD gfx950

### Warp Layout Chain (256×256 block, 512 threads = 8 warps)

```
T.gemm_v2(A_shared, B_shared, C_local)
  → GemmMFMA.lower() (tilelang/tileop/gemm/gemm_mfma.py)
    → policy.compute_warp_partition(M, N, block_size)  (src/op/gemm.cc)
      → m_warp=4, n_warp=2 (Square policy)
      → warp_row_tiles=64, warp_col_tiles=128
    → MatrixCoreIntrinEmitter (tilelang/intrinsics/mfma_macro_generator.py)
      → warp_rows=4, warp_cols=8 (MFMA 16×16 tiles per warp)
```

### Key Emitter Methods

| Method | What it does | Key offset |
|--------|-------------|------------|
| `extract_thread_binding(tid)` | tid → (lane_id, warp_n, warp_m) | `warp_m = (tid//64) % m_warp` |
| `ldmatrix_a(A_local, A_shared, ki)` | Shared → registers for A | `l = warp_m * 64 + i * 16` |
| `ldmatrix_b(B_local, B_shared, ki)` | Shared → registers for B | `r = warp_n * 128 + j * 16` |
| `mfma(A_local, B_local, C_local)` | MFMA compute | `for kp, i, j in grid(k_pack, warp_rows, warp_cols)` |
| `stmatrix(C_local, C_buf)` | Registers → output | Same offset pattern as ld |
| `make_mfma_store_layout` | Fragment layout: `forward_thread(i,j)→tid`, `forward_index(i,j)→local_id` |  |

### Index Map for A (non-transposed, bf16, k_dim=32)

`shared_16x32_to_local_64x8_layout_A`: 16×32 micro-tile, 64 threads, 8 elements each.
```
reverse_index_map(tx, local_id):
  row = tx % 16              # which row in 16×32 tile
  col = (tx // 16) * 8 + local_id  # which col (4 groups of 8)
```

## Completed Optimizations

### 1. Buffer Resource Hoisting (`HoistBufferResource` pass)

**Files**: `tilelang/transform/hoist_buffer_resource.py`, `src/tl_templates/hip/copy.h`, `src/target/codegen_hip.cc`, `src/op/builtin.h/.cc`

**What**: `cp_async_gs_lds<16>` calls `make_wave_buffer_resource()` (4× readfirstlane) on every invocation. We hoist it to kernel entry:
```cpp
auto __rsrc_A = make_wave_buffer_resource((const void*)(A));
uint32_t __base_A = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)(A));
// Then each load uses pre-computed rsrc + base:
tl::cp_async_gs_lds_with_rsrc<16>(lds_ptr, global_ptr, __rsrc_A, __base_A);
```

**How**:
- Python pass collects `ptx_cp_async_lds` calls, groups by buffer
- Creates `AttrStmt("buffer_resource_var")` and `AttrStmt("buffer_base_var")` wrappers
- Rewrites calls to `ptx_cp_async_lds_rsrc` with 5 args (dst, src, bytes, rsrc, base)
- Codegen handles the new AttrStmts and intrinsic

**Why AttrStmt not LetStmt**: `func.with_body()` calls `PrimFunc()` constructor → `IsPureFunction()` → segfaults on custom `kOpaque` ops in LetStmt values because static `OpAttrMap` is stale.

**Why `ir_transform` not `PyStmtExprMutator`**: `PyStmtExprMutator` corrupts the IR tree (returns single `DeclBuffer` node instead of full body). `stmt_functor.ir_transform` is the correct API for TIR rewrites.

### 2. AMD Async Wait Count Fix (same pass)

**What**: AMD has no commit groups. Each `buffer_load_dwordx4 ... lds` is tracked individually by `vmcnt`. `cp_async_wait<1>` emits `s_waitcnt vmcnt(1)` but should be `vmcnt(8)` (1 group × 8 loads/group).

**How**:
- `_get_loads_per_group()`: finds the For loop containing `async_commit_queue_scope`, counts ALL async loads in one iteration (including loads outside the commit scope but in the same loop body)
- `_fix_amd_wait_counts()`: replaces `async_wait_inflight_count` N with N × loads_per_group

**Gotcha**: A copies are OUTSIDE the commit scope in TIR, but NVIDIA's commit groups everything since last commit. So must count at loop-body level, not commit-scope level.

### 3. Scattered Warp Layout

**File**: `tilelang/intrinsics/mfma_macro_generator.py`

**What**: Instead of each warp handling one contiguous block (64×128), scatter into sub-tiles (32×64 each, 4 sub-tiles per warp):
```
Contiguous (m_warp=4, n_warp=2):     Scattered:
0 0 4 4   (each cell=64×64)          0 4 0 4  (each cell=32×64)
1 1 5 5                              1 5 1 5
2 2 6 6                              2 6 2 6
3 3 7 7                              3 7 3 7
                                     0 4 0 4
                                     1 5 1 5
                                     2 6 2 6
                                     3 7 3 7
```

**How**: `scattered_warp=True` flag changes offset formulas:
```python
# _scattered_m_offset(warp_m, i):
sub_rows = warp_rows // 2
sub_row_tiles = warp_row_tiles // 2
offset = (warp_m + (i // sub_rows) * block_row_warps) * sub_row_tiles + (i % sub_rows) * M_DIM

# Same pattern for _scattered_n_offset
```

Affects: `ldmatrix_a`, `ldmatrix_b`, `stmatrix`, `forward_thread`, `forward_index`. `block_row_warps` and `block_col_warps` unchanged.

### 4. Pingpong 4-Cluster Compute Schedule

**File**: `tilelang/tileop/gemm/gemm_mfma.py` (`_gemm_ssr`)

**What**: Split 32 MFMAs per ki into 4 clusters of 8 MFMAs, with shared→local loads distributed:
```
Cluster 0: ldA[0:sub_m], ldB[0:sub_n], mfma(sub_m=0, sub_n=0)  // 2×4=8 MFMAs
Cluster 1: ldB[sub_n:],              mfma(sub_m=0, sub_n=1)  // 2×4=8 MFMAs
Cluster 2: ldA[sub_m:],              mfma(sub_m=1, sub_n=0)  // 2×4=8 MFMAs
Cluster 3:                           mfma(sub_m=1, sub_n=1)  // 2×4=8 MFMAs
```

**Condition**: `scattered_warp and warp_rows >= 4 and warp_cols >= 4`

**New emitter methods**: `ldmatrix_a_subtile`, `ldmatrix_b_subtile`, `mfma_subtile` — operate on sub-ranges of i/j.

## TODO

### G2S Copy Interleaving (Next Step)

Currently all 8 G2S copies (4 for A + 4 for B) happen at the top of each k-iteration. Want to distribute them across the 4 compute clusters:
```
Cluster 0: ldA_sub0, ldB_sub0, G2S_A[0:2], mfma(0,0)
Cluster 1: ldB_sub1,           G2S_A[2:4], mfma(0,1)
Cluster 2: ldA_sub1,           G2S_B[0:2], mfma(1,0)
Cluster 3:                     G2S_B[2:4], mfma(1,1)
```

This allows more aggressive `s_waitcnt` — each cluster only needs to wait for its 2 G2S loads instead of all 8.

**Approach**: Write a lightweight post-pipeline Python pass that:
1. Finds G2S copy loops (cp_async_gs_lds_rsrc in unrolled for loops)
2. Splits them into 4 groups (2 loads each)
3. Moves each group to the corresponding cluster gap

### Triton BlockPingpong Reference

See `triton/third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp`:
- `sliceDot()`: splits dot into N pieces via `MemDescSubsliceOp` + cloned dots
- `transformFourPPClusters()`: interleaves 4 mem+dot cluster pairs
- Uses `s_setprio 0/1` + `cond_barrier` for warp synchronization
- Uses `sched_barrier(0)` to prevent compiler reordering across cluster boundaries

### Future: `s_setprio` and `sched_barrier`

After G2S interleaving works, add:
- `s_setprio 1` at cluster start / `s_setprio 0` at cluster end — prevents warp preemption during MMA
- `s_sched_barrier 0` at cluster boundaries — prevents LLVM backend from reordering across clusters
- `cond_barrier` for asymmetric warp synchronization (two warps on same SIMD pingpong between compute and memory clusters)
