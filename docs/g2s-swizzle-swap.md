# G2S Copy: buffer_load_b128 ... lds on AMD gfx950+

## 1. 目标

在 AMD ROCm (gfx950+) 上，使用硬件 `buffer_load_b128 ... lds` 指令实现 Global → Shared Memory 的 truly async copy，跳过 VGPR 中转。

**硬件行为**：
- 每条指令：64 lanes × 16 bytes = 1024 bytes
- LDS 写入地址：`m0 + lane_id * 16`（硬件强制，连续）
- Global 读取地址：`rsrc.base + soffset + voffset`（可以任意）
- 由 `vmcnt` 追踪（truly async，数据不经过 VGPR）

**前提**：LDS 写入地址必须按 `lane_id * N` 连续排列。但 tilelang 的 swizzle layout 会把 XOR 置换施加在 store 侧，导致 LDS 地址非连续。因此需要先把 swizzle 从 store 侧交换到 load 侧。

## 2. 已完成的工作

### 2.1 Swizzle 交换（lower_tile_op.cc）✅

**文件**：`src/transform/lower_tile_op.cc` — `VisitStmt_(const BufferStoreNode *op)`

**问题**：layout 同时做了 reshape + swizzle（`[256, 64]` → `[1, 32, 512]`），逐维度 delta 法不工作。

**解决方案**：Flatten-space delta 法。在 1D 空间中做差，reshape 项自动对消：

```
flat_swizzled  = flatten(layout->Forward(indices), new_buffer->shape)
flat_original  = flatten(indices, buffer->shape)
flat_delta     = flat_swizzled - flat_original = swizzle(col) - col
```

然后：
- **Store 侧**：`swizzled_store[-1] - flat_delta` → 去掉 swizzle，只保留 reshape
- **Load 侧**：`load[-1] + flat_delta` → swizzle 挪到 global 读取

**结果**：LDS 写入地址变为 `threadIdx.x * 8`（bf16 元素）= `threadIdx.x * 16` 字节，连续。验证正确。

### 2.2 cp_async_gs 改用 buffer_load_b128 ... lds ✅

**文件**：`src/tl_templates/hip/copy.h`

保持 `cp_async_gs<16>(lds_ptr, global_ptr)` 的函数签名不变，内部实现从 VGPR 中转改为 truly async：

```cpp
// 旧实现（VGPR 中转）
*(uint4 *)lds_base_ptr = *(const uint4 *)global_base_ptr;

// 新实现（truly async，bypasses VGPRs）
uint32_t lds_m0 = readfirstlane(lds_base_ptr);     // wave-uniform LDS offset
auto rsrc = make_wave_buffer_resource(global_ptr);   // lane 0 的 global 地址
uint32_t voffset = my_global_lo - base_global_lo;    // per-lane delta
s_mov_b32 m0, lds_m0;
llvm_amdgcn_raw_buffer_load_lds(rsrc, 0, 16, voffset, 0, 0, 0);
```

不需要修改 IR 或 codegen——签名一样，只是模板内部变了。

### 2.3 vmcnt 修复（codegen_hip.cc）✅

**文件**：`src/target/codegen_hip.cc` + `codegen_hip.h`

**问题**：NVIDIA 的 `ptx_wait_group(N)` 中 N 是 commit group 数。AMD 的 `vmcnt(N)` 中 N 是单条指令数。每个 commit group 包含 8 条 `buffer_load` 指令（4 for A + 4 for B），所以 `wait_group(1)` 应该映射到 `vmcnt(8)` 而不是 `vmcnt(1)`。

**解决方案**：在 codegen 中追踪每个 commit group 的实际指令数：

- `async_ops_since_commit_`：计数器，每次 `ptx_cp_async` 时加上 enclosing unrolled loop 的 trip count
- `ops_per_commit_group_`：`ptx_commit_group` 时记录并重置
- `ptx_wait_group(N)` 时发射 `vmcnt(N * ops_per_commit_group_)`

### 2.4 Barrier 改为 s_barrier（codegen_hip.cc）✅

**问题**：`__syncthreads()` 在 AMD 上展开为 `s_barrier` + `s_waitcnt vmcnt(0)`，后者会等完所有 load，杀掉 pipeline overlap。

**解决方案**：`PrintStorageSync` 中将 `__syncthreads()` 替换为 `__builtin_amdgcn_s_barrier()`（只发 `s_barrier`，不带 `vmcnt(0)`），vmcnt 由我们的 `cp_async_wait` 显式控制。

## 3. 当前卡点

### 编译器自动插入 vmcnt(0) ❌

**现象**：即使去掉了 `__syncthreads()`，编译器仍然在 LDS 读取（`ds_read_b128`，load shared → registers for MFMA）之前自动插入 `s_waitcnt vmcnt(0)`。

```asm
buffer_load_dwordx4 v130, s[0:3], 0 offen lds   ; 最后一条 async G2S load
s_waitcnt vmcnt(8)                                ; 我们的 cp_async_wait（正确）
s_barrier                                         ; 我们的 barrier（正确）
s_waitcnt vmcnt(0)                                ; ← 编译器自动插入！
ds_read_b128 ...                                  ; LDS 读取 for MFMA
```

**原因**：`llvm.amdgcn.raw.buffer.load.lds` intrinsic 有 LDS write 的 memory side-effect（通过 `as3_uint32_ptr` 参数）。编译器看到后续有 `ds_read` 从 LDS 读取，保守地认为可能 alias（不知道是双缓冲的不同半区），插入 `vmcnt(0)` 确保 LDS 写完再读。

**插入位置**：在 tilelang 生成的 host_kernel.cu 里，wait + barrier 之后紧接着两个循环：先 `*(uint4*)(A_local...) = *(uint4*)(buf_dyn_shmem + ...)`（A 的 LDS 读），再 `*(uint4*)(B_local...) = *(uint4*)(buf_dyn_shmem + ...)`（B 的 LDS 读）。汇编中编译器在**第一个**从 `buf_dyn_shmem` 的 LDS 读之前插 `s_waitcnt vmcnt(0)`，.loc 指向 B 那行（约第 50 行），即 pipeline 里第一次从 shared 读到 register 的那条语句。

**后果**：pipeline overlap 被完全杀掉——新 tile 的 8 条 load 刚发出去就被等完了，compute 和 load 完全串行。

### 对比：HipKittens 为什么不会自动插 vmcnt(0)

HipKittens 同样用 `llvm.amdgcn.raw.buffer.load.lds`（见 `include/ops/warp/memory/tile/global_to_shared.cuh` 约 299–306 行），且也传 `(as3_uint32_ptr)0`，LDS 实际地址只通过 `m0` 传。但汇编里只有手写的 `s_waitcnt vmcnt(4)`/`vmcnt(6)`，**没有**在 `ds_read_b128` 前多出 `s_waitcnt vmcnt(0)`。可能原因：

1. **LDS 读的“来源”在编译器眼里不同**  
   - **Tilelang**：LDS 读是**直接**对同一块 `buf_dyn_shmem` 的 C++ 解引用：`*(uint4*)((bfloat16_t*)buf_dyn_shmem + offset)`，且 `cp_async_gs` 的调用处传入的就是 `&buf_dyn_shmem[...]`。内联后编译器能关联“写 LDS 的 intrinsic”和“从 `buf_dyn_shmem` 读”是同一块共享内存，于是保守在第一次 LDS 读前插 `vmcnt(0)`。  
   - **HipKittens**：LDS 读在 `load(B_tile_0, st_subtile_b)` 等里，来自 `shared_to_register.cuh`，地址是 `&Bs[0][0].data[0]` 等**另一套 C++ 对象** + swizzle 计算，最终变成 ds_read 用的 v138、v139 等寄存器。intrinsic 侧只看到 `(as3_uint32_ptr)0`，编译器难以把“写 LDS 的 intrinsic”和“从 Bs[x][y].data 算出来的地址”当成同一块内存，alias 分析可能判定不 alias，因此不插 vmcnt(0)。

2. **控制流 / 代码布局**  
   HipKittens 里：`G::load`（buffer_load_lds）→ `asm volatile("s_waitcnt vmcnt(6)")` → `s_barrier` → 之后才到循环里的 `load(..., st_subtile)`（ds_read 在 .LBB0_9）。wait 和 LDS 读在不同 BB，且中间有**手写 inline asm**。插入 vmcnt 的 pass 可能不会在“已经有一条 opaque 的 wait 在路径上”时再在 LDS 读前插 vmcnt(0)。Tilelang 里 wait 是 `tl::cp_async_wait<8>()` 内联成的 asm，但 LDS 读**紧接**在 wait/barrier 之后、同一线性块里对 `buf_dyn_shmem` 的直接访问，编译器更容易在“第一次 shared 读”前插 vmcnt(0)。

**可借鉴的改法**（需验证）：  
- 让“第一次从 shared 读到 register”的代码不直接写 `*(uint4*)(buf_dyn_shmem + ...)`，而是通过一层 **`__attribute__((noinline))` 的辅助函数**或**另一编译单元**做 LDS 读，使插入 vmcnt 的 pass 在当前函数里看不到“对同一块 shared 的读”，从而不插 vmcnt(0)。  
- 或像方案 B：对这部分 LDS 读用手写 ds_read inline asm，让编译器看不到对 shared 的 load 依赖。

### 可能的解决方案

#### 方案 A：`__builtin_amdgcn_sched_barrier(0)`
在 barrier 和 LDS 读取之间插入 sched_barrier，阻止编译器跨过它调度 waitcnt。不确定能否阻止编译器插入 vmcnt。

#### 方案 B：用 inline asm 做 LDS 读取
手写 `ds_read_b128` 的 asm，编译器看不到 LDS load 依赖就不会插 vmcnt。但改动量大，需要把所有 LDS → register 的 load 都改成 asm。

#### 方案 C：给 buffer_load_lds intrinsic 加 noalias 或 metadata
告诉编译器 async G2S 写入和后续 LDS 读取的地址不 alias。需要 LLVM 层面的支持。

## 4. 修改文件清单

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `src/transform/lower_tile_op.cc` | Flatten-space delta swizzle 交换 | ✅ 已完成 |
| `src/tl_templates/hip/copy.h` | `cp_async_gs<16>` 改用 `buffer_load_b128 ... lds` | ✅ 已完成 |
| `src/target/codegen_hip.cc` | vmcnt 计算（group→instruction count）+ barrier 改 `s_barrier` | ✅ 已完成 |
| `src/target/codegen_hip.h` | 新增 `async_ops_since_commit_`、`ops_per_commit_group_`、`loop_trip_counts_` | ✅ 已完成 |

## 5. 相关文件参考

| 文件 | 作用 |
|------|------|
| `src/transform/inject_ptx_async_copy.cc` | 将 `BufferStore(shared, BufferLoad(global))` 转为 `ptx_cp_async` IR |
| `src/transform/inject_pipeline.cc` | 软件 pipeline 编排（async_scope、commit/wait group） |
| `src/layout/gemm_layouts.cc` | `makeMatrixCoreSwizzleLayout` — XOR swizzle layout 定义 |
| `src/op/copy.cc` | Copy 操作的 lowering（`LowerNormalCopy`, `MakeSIMTLoop`） |
| `tilelang/engine/phase.py` | Pass 执行顺序 |
