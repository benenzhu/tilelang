# TileLang MI355X (gfx950) 测试状态

测试日期: 2026-02-20
硬件: AMD MI355X (gfx950, 256 CU, ROCm 7.2.0)
TileLang 版本: 0.1.7.post3+rocm.git2cc71cea

## 总览

| 测试集 | Passed | Failed | Skipped | Total |
|--------|--------|--------|---------|-------|
| AMD 专属 (`testing/python/amd/`) | 7 | 25 | 0 | 32 |
| 通用 (排除 runtime/transform/platform-specific/tilelibrary) | 280 | 37 | 269 | 586 |

## AMD 专属测试详情

### PASS (7)

全部是 **fp16→fp16 GEMM**，使用 MFMA 16x16x16 指令，基础路径正常：

```
test_gemm_f16f16f32[False-False-1]  ✓
test_gemm_f16f16f32[True-True-1]    ✓
test_gemm_f16f16f32[True-False-1]   ✓
test_gemm_f16f16f32[False-True-2]   ✓
test_gemm_f16f16f32[False-False-2]  ✓  (preshuffle)
test_gemm_f16f16f32[False-True-1]   ✓  (preshuffle)
test_gemm_f16f16f32[False-True-2]   ✓  (preshuffle)
```

### FAIL (25)

#### 1. MFMA intrinsic GEMM (9 failures)

文件: `test_tilelang_gemm_mfma_intrinsic.py`

| 测试 | 类型 | 失败模式 |
|------|------|---------|
| `[128-128-128-float16-float16-float32-False-True-1]` | fp16, trans_B | 计算错误 |
| `[128-256-256-float16-float32-float32-False-True-1]` | fp16→f32 | 计算错误 |
| `[128-256-256-float16-float32-float32-False-True-2]` | fp16→f32, k_pack=2 | 计算错误 |
| `[128-128-128-int8-int32-int32-False-True-1]` | int8 | 计算错误 |
| `[128-256-256-int8-int32-int32-False-True-1]` | int8 | 计算错误 |
| `[128-256-256-int8-int32-int32-False-True-2]` | int8, k_pack=2 | 计算错误 |
| `[128-256-256-int8-int32-int32-False-False-1]` | int8, NN | 计算错误 |
| `[128-256-256-int8-int32-int32-False-False-2]` | int8, NN, k_pack=2 | 计算错误 |
| `[128-128-128-float8_e4m3fnuz-float16-float32-False-True-1]` | fp8 | 计算错误 |

#### 2. MFMA preshuffle GEMM (4 failures)

文件: `test_tilelang_gemm_mfma_preshuffle.py`

| 测试 | 失败模式 |
|------|---------|
| `[256-256-512-float8_e4m3fnuz-...-True-1-True-False]` | fp8 preshuffle 计算错误 |
| `[256-256-512-float8_e4m3fnuz-...-False-1-True-False]` | fp8 preshuffle 计算错误 |
| `[256-256-512-float8_e4m3fnuz-...-True-2-True-False]` | fp8 preshuffle k_pack=2 |
| `[256-256-512-float8_e4m3fnuz-...-False-2-True-False]` | fp8 preshuffle k_pack=2 |

#### 3. 基础 GEMM (12 failures)

文件: `test_tilelang_test_amd.py`

| 测试组 | 失败数 | 失败模式 |
|--------|--------|---------|
| `test_gemm_f16f32f32_nt` (fp16 input, f32 output) | 4 | 99%+ 元素 mismatch |
| `test_gemm_bf16f32f32_nt` (bf16 input, f32 output) | 4 | 99%+ 元素 mismatch |
| `test_gemm_bf16bf16f32` (bf16 input, bf16 output) | 4 | 99%+ 元素 mismatch，部分输出全零 |

所有 4 种 layout 组合 (NN/NT/TN/TT + k_pack 1/2) 均失败。

## 通用测试失败详情 (37 failures)

### GEMM 正确性 (~20 failures)

与 AMD 测试同源，MFMA layout 在 gfx950 上不正确：

```
test_gemm_f16f16f32_nn              FAILED
test_gemm_bf16bf16f32_nn            FAILED
test_gemm_f32f32f32_nn              FAILED
test_gemm_f32f32f32_nt              FAILED
test_pad_f16f16f32_nn               FAILED
test_gemm_f16f16f32_nn_kernel_jit   FAILED
test_gemm_jit_kernel (callback)     FAILED
test_gemm_jit_kernel (cython)       FAILED
test_cython_dynamic_shape           FAILED
test_cython_dynamic_shape_with_out_idx  FAILED
test_matmul_int_variable            FAILED
test_matmul_float_variable          FAILED
test_gemm_jit_kernel (tvm_ffi)      FAILED
test_tvm_ffi_dynamic_shape          FAILED
test_par_compile                    FAILED
test_jit2_gemm                      FAILED
```

### Sparse matmul (5 failures)

```
test_block_sparse_matmul_global  (all_of)   FAILED
test_block_sparse_matmul_shared  (all_of)   FAILED
test_block_sparse_matmul_global  (any_of)   FAILED
test_block_sparse_matmul_shared  (any_of)   FAILED
test_block_sparse_matmul_local   (any_of)   FAILED
```

### cumsum (3 failures)

```
test_cumsum_smem        FAILED
test_cumsum_fragment     FAILED
test_cumsum_region_2d    FAILED
```

### Pipeline/Scheduling (2 failures)

```
test_pipeline_order_stage     FAILED
test_blocksparse_matmul       FAILED
```

### Autotune (2 failures)

```
test_autotune_matmul                  FAILED
test_autotune_matmul_symbolic_m       FAILED
```

### Profiler (2 failures)

```
test_profiler                                  FAILED
test_profiler_dynamic_symbolic_correctness     FAILED
```

### Analysis (3 failures)

```
test_nested_pipelines           FAILED
test_mixed_pp                   FAILED
test_tiled_op_with_parallel     FAILED
```

### 其他 (2 failures)

```
test_debug_print_buffer_rocm_fp8    FAILED  (fp8 type 打印)
test_issue_1106                     FAILED
test_issue_96 (large + small)       FAILED  (pipeline matrix)
```

### Collection error (1)

```
test_tilelang_tilelibrary_gemm.py  ERROR  (import-time dtype mismatch)
```

## 根因分析

### 核心问题: gfx950 MFMA 指令差异

gfx950 的 MFMA 指令与 gfx942 (MI300X) 不同:

| 属性 | gfx942 (MI300X) | gfx950 (MI355X) |
|------|-----------------|-----------------|
| bf16 MFMA | `v_mfma_f32_16x16x16bf16_1k` (K=16) | `v_mfma_f32_16x16x32_bf16` (K=32) |
| 输入 vector | 4 个 bf16 元素 | 8 个 bf16 元素 |
| k_pack | 与 K=16 匹配 | 需要适配 K=32 |

fp16 MFMA 16x16x16 在两代硬件上相同，所以 fp16→fp16 测试通过。
bf16/int8/fp8 的 K 维度和 vector 宽度在 gfx950 上翻倍，layout mapping 需要适配。

### 失败分布

```
MFMA layout 问题 (bf16/int8/fp8)  →  ~45 个测试 (覆盖 AMD + 通用)
Sparse/cumsum/pipeline (独立问题)  →  ~10 个测试
连锁失败 (autotune/profiler)      →  ~6 个测试
Analysis pass 断言               →  3 个测试
```

## 建议修复顺序

1. **bf16 MFMA 16x16x32 layout 适配** — 修好后预计 ~30 个测试变绿
   - `tilelang/intrinsics/mfma_macro_generator.py` — layout mapping
   - `tilelang/intrinsics/mfma_layout.py` — gfx950 layout specs
   - `src/tl_templates/hip/gemm.h` — MfmaBf16Traits

2. **int8 MFMA layout** — 类似 bf16 的适配，预计 ~8 个测试

3. **fp8 支持** — 可能需要额外的 gfx950 intrinsic，预计 ~5 个测试

4. **Sparse/cumsum/pipeline** — 独立问题，逐个排查

5. **Analysis pass** — nested loop checker 断言，可能是 TIR 结构差异
