# G2S Copy: buffer_load_b128 ... lds on AMD gfx950+

## 1. 目标

在 AMD ROCm (gfx950+) 上，使用硬件 `buffer_load_b128 ... lds` 指令实现 Global → Shared Memory 的 truly async copy，跳过 VGPR 中转。

**硬件行为**：
- 每条指令：64 lanes × 16 bytes = 1024 bytes
- LDS 写入地址：`m0 + lane_id * 16`（硬件强制，连续）
- Global 读取地址：`rsrc.base + soffset + voffset`（可以任意）
- 由 `vmcnt` 追踪（truly async，数据不经过 VGPR）

**前提**：LDS 写入地址必须按 `lane_id * N` 连续排列。但 tilelang 的 swizzle layout 会把 XOR 置换施加在 store 侧，导致 LDS 地址非连续。因此需要先把 swizzle 从 store 侧交换到 load 侧。

## 2. 已完成的工作（精简）

| # | 内容 | 文件 |
|---|------|------|
| 2.1 | **Swizzle 交换**：Flatten-space delta 法，把 XOR swizzle 从 LDS store 侧移到 global load 侧，使 LDS 写入地址连续 | `src/transform/lower_tile_op.cc` |
| 2.2 | **cp_async_gs 改用 buffer_load_b128…lds**：函数签名不变，内部从 VGPR 中转改为 truly async | `src/tl_templates/hip/copy.h` |
| 2.3 | **vmcnt 修复**：`ptx_wait_group(N)` 映射为 `vmcnt(N × ops_per_group)`，追踪每个 commit group 的实际指令数 | `src/target/codegen_hip.cc` + `.h` |
| 2.4 | **Barrier 改 s_barrier**：去掉 `__syncthreads()` 隐含的 `vmcnt(0)`，改用 `__builtin_amdgcn_s_barrier()` | `src/target/codegen_hip.cc` |
| 2.5 | **共享内存不合并 + 静态声明**（见下文 §3）| `codegen_hip.cc`, `phase.py`, `wrapper.py`, `lower_device_kernel_launch.cc` |

## 3. 编译器自动插入 vmcnt(0) — 问题与解决

### 3.1 现象

即使用了 `s_barrier`（不带 `vmcnt(0)`），编译器仍在 `ds_read_b128`（S2R load for MFMA）之前自动插入 `s_waitcnt vmcnt(0)`，彻底杀掉 pipeline overlap。

```asm
buffer_load_dwordx4 ... lds   ; 最后一条 async G2S load
s_waitcnt vmcnt(8)             ; 我们的 cp_async_wait（正确）
s_barrier                      ; 我们的 barrier（正确）
s_waitcnt vmcnt(0)             ; ← 编译器自动插入！
ds_read_b128 ...               ; LDS 读取 for MFMA
```

### 3.2 根因分析

LLVM 后端 `SIInsertWaitcnts` pass 的判断链：

1. `buffer_load_b128…lds` intrinsic 的 LDS 写副作用通过 dummy `(as3_uint32_ptr)0` 表达
2. 后续 `ds_read` 从 shared memory 读取
3. 如果编译器能把两者关联到**同一个底层对象**，就判定 MayAlias → 插入 `vmcnt(0)`

关键在于 LLVM 的 **pointer provenance / alias analysis**：

| 声明方式 | LLVM IR | Identified Object? | 与 null AS3 ptr 的关系 | 结果 |
|----------|---------|--------------------|-----------------------|------|
| `extern __shared__ uchar buf[]` | `external addrspace(3) global [0 x i8]` | ❌（外部链接，大小未知） | MayAlias | 插入 vmcnt(0) ❌ |
| `__shared__ uchar buf[131072]`（单个静态） | `internal addrspace(3) global [131072 x i8]` | ✅ 但仍是同一对象 | 实测仍 MayAlias | 插入 vmcnt(0) ❌ |
| 多个独立 `__shared__ T A[N]; __shared__ T B[M];` | 各自 `internal addrspace(3) global` | ✅ 各自独立 | NoAlias | 不插入 vmcnt(0) ✅ |

**结论**：只有**多个独立的 `__shared__` 声明**才能让 LLVM 判定每个缓冲区与 `buffer_load_lds` intrinsic 的 dummy null 指针不 alias。

### 3.3 HipKittens 为什么不受影响

HipKittens 有**两道保险**：

1. **Inline asm `ds_read_b128`**（`include/common/macros.cuh`）：S2R 读取用 `asm volatile("ds_read_b128 v[%0:%1], %2 offset:%3")`，LLVM 看不到显式的 LDS load 语义
2. **ptrtoint 切断 provenance**（`shared_to_register.cuh`）：地址先转为 `uint32_t` 整数再传给 asm

TileLang 不采用 inline asm ds_read 的原因：AMD 没有 PTX 这样的虚拟 ISA，inline asm 中的寄存器号直接进入最终机器码，register spill/重排会导致冲突。

### 3.4 解决方案：跳过共享内存合并 ✅

**核心思路**：不把所有 shared buffer 合并为单个 `buf_dyn_shmem`，让每个 buffer 保持独立的 `__shared__` 声明。

**修改文件**：

| 文件 | 修改 |
|------|------|
| `tilelang/engine/phase.py` | HIP 目标跳过 `MergeSharedMemoryAllocations` pass |
| `src/target/codegen_hip.cc` | `PrintStorageScope("shared.dyn")` 改为 `__shared__ __align__(1024)` （去掉 `extern`）；`VisitStmt_(AllocateNode)` 对 `shared.dyn` 发射固定大小数组 `buf[SIZE]` 而非 `buf[]` |
| `src/transform/lower_device_kernel_launch.cc` | 允许多个 `shared.dyn` 分配（累加大小，去掉 "only one" 断言） |
| `tilelang/jit/adapter/wrapper.py` | `TLHIPSourceWrapper.get_launch_smem_size()` 始终返回 0（静态共享内存不需要在 launch 时传大小） |

**生成代码对比**：

```cpp
// 修改前（合并为单个 extern 动态共享）
extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
// launch: <<<grid, block, 131072, stream>>>

// 修改后（每个 buffer 独立的静态共享）
__shared__ __align__(1024) bfloat16_t A_shared_dyn[32768];
__shared__ __align__(1024) bfloat16_t B_shared_dyn[32768];
// launch: <<<grid, block, 0, stream>>>
```

**效果**：编译器不再在 `ds_read` 前插入 `vmcnt(0)`，pipeline overlap 得以保持。

### 3.5 已排除的方案

| 方案 | 结论 |
|------|------|
| `__builtin_assume` 约束地址范围 | 不影响 `SIInsertWaitcnts` pass |
| `reinterpret_cast` 从 `buf_dyn_shmem` 派生 | provenance 不断，LLVM 追溯到同一 extern 对象 |
| 包装函数切断 provenance | inline 后等价于直接 cast；`noinline` 有 call overhead |
| 单个 `__shared__ buf[SIZE]`（静态但不分离） | 实测仍插入 vmcnt(0) |
| `__builtin_amdgcn_sched_barrier(0)` | 不阻止 waitcnt 插入 |

### 3.6 后续：共享内存复用

跳过合并意味着不同阶段（如 attention 中 QK → V）无法复用同一块 LDS。对 GEMM 无影响（双缓冲的 A/B 全程存活）。后续如需复用，可考虑：

- 不同 pipeline stage 之间本来有 `vmcnt(0)` + barrier，跨阶段没有 vmcnt 问题，可在 stage 边界重新分配
- 或：仅对 async G2S pipeline 阶段内的 buffer 保持独立声明，其他继续用 `buf_dyn_shmem`

## 4. 修改文件清单

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `src/transform/lower_tile_op.cc` | Flatten-space delta swizzle 交换 | ✅ |
| `src/tl_templates/hip/copy.h` | `cp_async_gs<16>` 改用 `buffer_load_b128 ... lds` | ✅ |
| `src/target/codegen_hip.cc` | vmcnt 计算 + barrier 改 `s_barrier` + 静态 `__shared__` 声明 | ✅ |
| `src/target/codegen_hip.h` | 新增 `async_ops_since_commit_` 等计数器 | ✅ |
| `tilelang/engine/phase.py` | HIP 跳过 `MergeSharedMemoryAllocations` | ✅ |
| `tilelang/jit/adapter/wrapper.py` | HIP launch smem_size=0 | ✅ |
| `src/transform/lower_device_kernel_launch.cc` | 支持多个 `shared.dyn` 分配（累加大小） | ✅ |

## 5. 相关文件参考

| 文件 | 作用 |
|------|------|
| `src/transform/inject_ptx_async_copy.cc` | 将 `BufferStore(shared, BufferLoad(global))` 转为 `ptx_cp_async` IR |
| `src/transform/inject_pipeline.cc` | 软件 pipeline 编排（async_scope、commit/wait group） |
| `src/layout/gemm_layouts.cc` | `makeMatrixCoreSwizzleLayout` — XOR swizzle layout 定义 |
| `src/op/copy.cc` | Copy 操作的 lowering（`LowerNormalCopy`, `MakeSIMTLoop`） |
| `src/transform/merge_shared_memory_allocations.cc` | 共享内存合并（HIP 上跳过） |
