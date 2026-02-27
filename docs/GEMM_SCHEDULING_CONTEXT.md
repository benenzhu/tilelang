# TileLang GFX950 GEMM Fine-Grained S2R/MFMA Interleaving

## 1. 目标

在 TileLang 为 AMD GFX950 (MI300X) 生成的 GEMM kernel 中，实现**精细的 S2R (Shared→Register) load 与 MFMA compute 的交错调度**，
以 hide memory latency、提高 MFMA 利用率。参考目标是 HipKittens 手写 kernel 的调度质量。

## 2. 硬件与 Kernel 参数

| 参数 | 值 |
|------|------|
| Target | AMD GFX950 (MI300X) |
| MFMA | `v_mfma_f32_16x16x32_bf16` |
| Block tile | 256×256×64 (M×N×K) |
| Warp size | 64 |
| Threads per block | 512 (8 warps: 4 warp_m × 2 warp_n) |
| warp_rows | 4 (每 warp 处理 4 个 16-row tile) |
| warp_cols | 8 (每 warp 处理 8 个 16-col tile) |
| k_pack | 2 |
| micro_size_k | 32 (bf16, GFX950 特有) |
| S2R load 单元 | `ds_read_b128` (16 bytes = 8 bf16) |
| in_dtype | bfloat16, accum_dtype = float32 |
| Layout | A: (M,K), B: (N,K) transposed (NT layout) |

### 每 warp 的寄存器用量

- **A_local**: `warp_rows × local_size_a × k_pack = 4 × 4 × 2 = 32` 个 bf16 slots → 每 row 16 bf16 = 2×ds_read_b128
- **B_local**: `local_size_b × k_pack = 4 × 2 = 16` 个 bf16 slots → 1 col = 2×ds_read_b128 (每轮复用)
- **C_local**: `warp_rows × warp_cols × local_size_out = 4 × 8 × 4 = 128` floats

### ds_read / MFMA 数量（单次 ki 迭代）

- **S2R loads**: 4 A rows × 2 + 8 B cols × 2 = **24** ds_read_b128
- **MFMA**: 4 rows × 8 cols × 2 kp = **64** v_mfma

## 3. 当前调度方案（已验证可工作）

### 3.1 总体流程

```
┌─ G2S: cp_async A(4), B(4) + commit + wait + s_barrier ─┐
│                                                          │
│  sched_barrier(0)                                        │
│  ┌─ Preamble: A row0(2), row1(2), B col0(2) ─┐ 6 ds_read│
│  sched_barrier(0)                                        │
│  setprio_hi(C, row0, row1)                               │
│  ┌─ Round 0a: MFMA row0×col0, row1×col0 ─┐  4 MFMA     │
│  setprio_lo(C, row0, row1)                               │
│  s_barrier + sched_barrier(0)                            │
│  ┌─ Load A row2(2), sched_barrier, row3(2) ─┐ 4 ds_read │
│  sched_barrier(0)                                        │
│  setprio_hi(C, row2, row3) + sched_barrier(0)            │
│  ┌─ Round 0b: MFMA row2×col0, row3×col0 ─┐  4 MFMA     │
│  sched_barrier(0) + setprio_lo(C, row2, row3)            │
│  s_barrier + sched_barrier(0)                            │
│  ┌─ Round 1: B col1(2) + 8 MFMA ─────────┐  setprio 1/0│
│  s_barrier + sched_barrier(0)                            │
│  ┌─ Round 2: B col2(2) + 8 MFMA ─────────┐              │
│  sched_barrier(0)                                        │
│  ┌─ Round 3: B col3(2) + 8 MFMA ─────────┐              │
│  sched_barrier(0)                                        │
│  ...                                                     │
│  ┌─ Round 7: B col7(2) + 8 MFMA ─────────┐              │
└──────────────────────────────────────────────────────────┘
```

### 3.2 每阶段详细说明

| 阶段 | S2R loads | MFMA | 保护机制 |
|------|-----------|------|----------|
| **Preamble** | A row0-1 + B col0 = 6 | 0 | `sched_barrier` 前后 |
| **Round 0a** | 0 | rows 0-1 × col 0 = 4 | `tl_setprio_hi/lo` (asm volatile + VGPR dep) |
| **A load 2-3** | A row 2-3 = 4 | 0 | `s_barrier` + `sched_barrier` 每行之间 |
| **Round 0b** | 0 | rows 2-3 × col 0 = 4 | `tl_setprio_hi/lo` (asm volatile + VGPR dep) |
| **Round 1** | B col1 = 2 | 4 rows × col1 = 8 | `s_setprio(1/0)` + `s_barrier` |
| **Rounds 2-7** | B colN = 2 × 6 = 12 | 4 rows × 6 cols = 48 | `sched_barrier` between load/MFMA |
| **合计** | 24 ds_read_b128 | 64 v_mfma | — |

### 3.3 当前方案的问题

Round 0 被拆成 0a + 0b，中间需要 4 条 ds_read 来加载 A row 2-3。
这意味着 **首批 MFMA 只有 4 条**（rows 0-1），然后中断去做 load，再做 4 条 MFMA（rows 2-3）。
理想情况下希望首批 MFMA 更多（如 8 条），这样 MFMA pipeline 可以更充分地 hide load latency。

## 4. 待实现的优化方向

### 4.1 Preamble 改进思路

**核心想法**：改变 Preamble 的 load 策略，让首批 MFMA 从 4 条变成 8 条。

当前 Preamble (6 ds_read → 4 MFMA):
```
Load A row0 full (2 ds_read)
Load A row1 full (2 ds_read)
Load B col0 full (2 ds_read)
→ 只能做 rows 0-1 × col 0 = 4 MFMA
```

可能的替代方案 — **加载 A 全部 4 rows + B col0** (10 ds_read → 8 MFMA):
```
Load A row0 full (2 ds_read)
Load A row1 full (2 ds_read)
Load A row2 full (2 ds_read)
Load A row3 full (2 ds_read)
Load B col0 full (2 ds_read)
→ 可以做 rows 0-3 × col 0 = 8 MFMA (不拆分 Round 0)
```
优势: 首批连续 8 MFMA, 无中断；Round 0 不需要拆分 0a/0b
代价: Preamble 从 6 增到 10 ds_read (多出 4 条 load 的启动延迟)

或者 — **部分 kp 加载** (仍 6 ds_read → 8 MFMA):
```
Load A row0 kp=0 (1 ds_read)
Load A row1 kp=0 (1 ds_read)
Load B col0-col3 各 kp=0 (4 × 1 ds_read)
→ 可以做 rows 0-1 × cols 0-3 × kp=0 = 8 MFMA
```
但这需要 B_local 能同时持有 4 列（当前只能持有 1 列），
或者改成 4 次单条 B load + MFMA 的交错模式。

**这部分是下一步要探索和实现的。**

### 4.2 Pingpong Buffering (更远期)

参考 HipKittens 中的 pingpong 模式，在 MFMA rounds 之间插入 G2S copy，
实现 compute 和 global memory transfer 的 overlap。
这需要在 Round 间的 `s_barrier` 位置精确地插入 cp_async 调用。

## 5. 关键代码文件

### 5.1 Lower 代码（TIR 代码生成）

**`tilelang/tileop/gemm/gemm_mfma.py`** — `GemmMFMA.lower()` → `_gemm_ssr()`

这是 SS GEMM 变体的核心 lower 函数。当前已实现 8-round 展开 + 调度屏障：
- `tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", ...)` 插入 sched_barrier(0)
- `tir.call_extern("int32", "tl_setprio_hi", C_buf.data, off1, off2)` 插入 asm volatile setprio
- `tir.call_extern("int32", "__builtin_amdgcn_s_barrier")` 插入 s_barrier
- `tir.call_extern("int32", "__builtin_amdgcn_s_setprio", ...)` 插入简化版 setprio

方法列表（在 `MatrixCoreIntrinEmitter` 上）:
- `ldmatrix_a_single(A_local, A_region, ki, row_idx)` — 加载 A 的单行
- `ldmatrix_b_single(B_local, B_region, ki, col_idx)` — 加载 B 的单列
- `mfma_cell(A, B, C, ki, row, col)` — 单个 (row, col) 的 k_pack 次 MFMA
- `mfma_col_slice(A, B, C, ki, col)` — 全部 rows × 单个 col 的 MFMA
- `mfma_slice(A, B, C, ki, row)` — 单个 row × 全部 cols 的 MFMA

### 5.2 MFMA 宏生成器

**`tilelang/intrinsics/mfma_macro_generator.py`** — `MatrixCoreIntrinEmitter`

定义了所有 S2R load 和 MFMA 的 TIR 宏展开逻辑。

### 5.3 C++ 模板头文件

**`src/tl_templates/hip/gemm.h`** — 包含：
- `MfmaTraits` / `GemmTensorOp` (C++ template GEMM 实现)
- `tl_setprio_hi(float* C, int off1, int off2)` — asm volatile s_setprio 1 + VGPR dep
- `tl_setprio_lo(float* C, int off1, int off2)` — asm volatile s_setprio 0 + VGPR dep

这两个函数在全局命名空间，TIR 通过 `call_extern` 直接调用。
`"+v"` / `"v"` 约束防止 LLVM 重排或消除 setprio 指令。

**`src/tl_templates/hip/common.h`** — 基础类型定义：
- `float32x4` = `__attribute__((__vector_size__(4 * sizeof(float)))) float`
- `bfloat16x8_vec` 等

### 5.4 示例 / 测试文件

**`examples/gemm/example_gemm.py`** — 主测试文件
- `matmul_nt()`: 256×256×64 block, transpose_B=True, k_pack=2
- 运行: `python examples/gemm/example_gemm.py`

**`examples/gemm/compile.sh`** — 手动编译缓存中的 host_kernel.cu
```bash
hipcc -std=c++17 -fPIC --save-temps -g --offload-arch=gfx950 --shared \
  "${CACHE_DIR}/host_kernel.cu" \
  -I/A/tilelang/3rdparty/composable_kernel/include \
  -I/A/tilelang/3rdparty/../src \
  -o "${CACHE_DIR}/kernel_lib.so"
```

**`examples/gemm/pure.sh`** — 从 .cu 提取纯汇编 (.spure.s)

### 5.5 参考文件

**HipKittens 参考 kernel**: `/root/learn-hip/HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp`
- 手写 GEMM，展示了 pingpong buffering 和精细的调度控制

**TileLang 缓存目录**: `/root/.tilelang/cache/<hash>/host_kernel.cu`
- TileLang 生成的 C++/HIP kernel 源码

## 6. 调度屏障机制详解

### 6.1 `__builtin_amdgcn_sched_barrier(0)`

编译器调度屏障：阻止 LLVM 跨越此点重排指令。
参数 0 = 禁止任何指令跨越。
在 TIR 中通过 `tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))` 发射。

### 6.2 `__builtin_amdgcn_s_barrier()`

Workgroup 级硬同步屏障，所有 wave 必须到达后才继续。
主要用于 G2S copy 后的同步，以及 Round 间的分隔。

### 6.3 `asm volatile("s_setprio N" : "+v"/v"(...) : : "memory")`

Wave 优先级设置。VGPR 约束 (`"+v"` 读写 / `"v"` 只读) 的作用：
- **防止 LLVM 将 setprio 指令优化掉**（volatile + memory clobber）
- **创建对 C_local 特定 tile 的显式寄存器依赖**，阻止 LLVM 跨越此点移动 MFMA 或 load

C_local 偏移计算：`offset = row_idx × warp_cols` (float32x4 单位)
- Row 0 → offset 0
- Row 1 → offset 8
- Row 2 → offset 16
- Row 3 → offset 24

### 6.4 `__builtin_amdgcn_s_setprio(N)`

简化版 setprio，无 VGPR 约束。用于 Round 1 等不需要强制寄存器依赖的场景。

## 7. 构建 & 测试流程

```bash
# TileLang 已从 /A/tilelang 以 dev 模式安装
# （注意：pip install -e . 在纯 ROCm 环境下可能因 CUDA 检测失败，
#  但 python import 可以正常工作，因为 build/ 已预构建）

# 运行测试
cd /A/tilelang
python examples/gemm/example_gemm.py

# 手动编译检查
bash examples/gemm/compile.sh

# 查看生成代码
# kernel.get_kernel_source() 或查看缓存目录下的 host_kernel.cu

# 查看汇编（纯净版）
bash examples/gemm/pure.sh
```

## 8. 生成代码结构（host_kernel.cu 主循环）

```cpp
// === G2S Prologue (k=0) ===
// cp_async A, B → shared
// s_barrier

for (int k = 0; k < 127; ++k) {
    // === G2S for k+1 ===
    s_barrier();
    cp_async A[k+1] → A_shared[(k+1)&1]
    s_barrier();
    cp_async B[k+1] → B_shared[(k+1)&1]
    cp_async_commit(); cp_async_wait<8>(); s_barrier();

    // === S2R + MFMA for k (interleaved) ===
    sched_barrier(0);

    // Preamble: A row0,1 + B col0
    A_local[0..15]  ← A_shared[k&1][row0]   // 2 ds_read
    A_local[16..31] ← A_shared[k&1][row1]   // 2 ds_read
    B_local[0..15]  ← B_shared[k&1][col0]   // 2 ds_read

    sched_barrier(0);
    tl_setprio_hi(C_local, 0, 8);
    // Round 0a: 4 MFMA (row 0-1 × col 0)
    tl_setprio_lo(C_local, 0, 8);

    s_barrier(); sched_barrier(0);
    // A row2, sched_barrier, A row3
    sched_barrier(0);
    tl_setprio_hi(C_local, 16, 24); sched_barrier(0);
    // Round 0b: 4 MFMA (row 2-3 × col 0)
    sched_barrier(0);
    tl_setprio_lo(C_local, 16, 24);

    s_barrier(); sched_barrier(0);
    // Round 1: B col1 load + s_setprio(1) + 8 MFMA + s_setprio(0)
    sched_barrier(0); s_barrier(); sched_barrier(0);

    // Rounds 2-7: [B colN load, sched_barrier, 8 MFMA, sched_barrier] × 6
}

// === Epilogue: last k iteration (no G2S) ===
```

## 9. 关键设计决策记录

1. **只改 SS GEMM**：其他变体 (SR/RS/RR) 暂不动，先把 SS 调好。
2. **Round 0 拆分为 0a/0b**：因为一次性加载 4 行 A 需要 8 条 ds_read，preamble 太长。
   拆分后 preamble 只需 6 ds_read，A rows 2-3 在 round 0a 的 MFMA 后面加载（hide latency）。
3. **asm volatile vs __builtin**：Round 0a/0b 必须用 asm volatile + VGPR 约束，否则 LLVM 会优化掉 setprio。
   Round 1 可以用简化版 `__builtin_amdgcn_s_setprio`，因为有 `sched_barrier` 保护。
4. **B_local 大小只能放 1 列**：每轮复用，所以 Round 2-7 的 pattern 是 "load B col → MFMA all rows"。
5. **sched_barrier(0) 放在每个 load/MFMA 边界**：这是防止 LLVM 重排的核心机制。
6. **s_barrier() 用于 Round 间分隔**：Round 0a→0b, 0b→1, 1→2 之间有 s_barrier，确保 G2S 数据可见。

## 10. 下一步 TODO

- [ ] **探索 Preamble 优化**：改变首批 load 策略，让首批连续 MFMA 从 4 条增加到 8 条
- [ ] **Pingpong buffering**：在 Round 间插入 G2S copy，实现 compute/transfer overlap
- [ ] **验证 `_Simplify` pass 不破坏 `call_extern` 顺序**：确保 TIR 优化不会移除或重排屏障
- [ ] **测试不同 block size**：当前只测了 256×256×64，需要验证其他配置
