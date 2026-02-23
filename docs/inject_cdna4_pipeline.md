# InjectCdna4Pipeline Pass 详解

## 一、目标

将 TileLang 标准 pipeline（先集中发所有 G2S，再集中做所有计算）变换为 HipKittens 风格的交错调度（G2S 穿插在 MFMA 计算阶段之间），以实现 G2S 内存传输和 MFMA 计算的重叠。

## 二、运行位置

在 `phase.py` 的 `OptimizeForTarget` 中，位于：
- **之后**：`InjectSoftwarePipeline` → `LowerOpaqueBlock` → `Simplify`
- **之前**：`NarrowDataType` → `FlattenBuffer` → ... → `ThreadSync` → `InjectPTXAsyncCopy`

即此时 IR 已经做完了软件流水线（double buffering），但还没有做 buffer 展平、向量化、线程同步注入等。

## 三、输入 IR 结构

pass 的输入是 `InjectSoftwarePipeline` 之后的 IR，结构如下：

```
prologue:
  async_scope { G2S A[0] }           ← 把 A 的第一个 tile 加载到 shared[0]
  async_commit { G2S B[0] }          ← 把 B 的第一个 tile 加载到 shared[0]

k-loop: for k in range(127):         ← 流水线主循环
  ┌─ G2S 部分 ─────────────────────────────────────────────┐
  │ async_scope {                                           │
  │   for i_1 in unroll(4):                                 │
  │     A_shared[(k+1)%2, ...] = A[..., k*64 + ... + 64]   │← 加载 A 的 k+1 tile
  │ }                                                       │
  │ async_commit {                                          │
  │   for i_1 in unroll(4):                                 │
  │     B_shared[(k+1)%2, ...] = B[..., k*64 + ... + 64]   │← 加载 B 的 k+1 tile
  │ }                                                       │
  └─────────────────────────────────────────────────────────┘
  ┌─ consumer 部分（async_wait 包裹）──────────────────────┐
  │ async_wait(inflight=1)  ← 等待 k 的 G2S 完成           │
  │   A_local = T.decl_buffer(...)                          │
  │   B_local = T.decl_buffer(...)                          │
  │                                                         │
  │   S2R A: for i_1, local_id in grid(4, 2):              │
  │     A_local[...] = A_shared[k%2, ...]    ← ds_read     │
  │                                                         │
  │   S2R B: for j, local_id in grid(8, 2):                │
  │     B_local[...] = B_shared[k%2, ...]    ← ds_read     │
  │                                                         │
  │   MFMA:  for kp, i_1, j in grid(2, 4, 8):             │
  │     tvm_mfma(B_local, j*2+kp, A_local, i_1*2+kp, ...) │
  └─────────────────────────────────────────────────────────┘

epilogue:
  async_wait(inflight=0)
  S2R + MFMA for last tile
```

**关键参数**（对于 256x256x64 bf16 GEMM, 512 threads）：
- `G2S A` = 4 条 `buffer_load_b128...lds`（每条 1024 bytes）
- `G2S B` = 4 条
- `S2R A` = 4 rows × 2 kpack = 8 条 `ds_read_b128`
- `S2R B` = 8 cols × 2 kpack = 16 条 `ds_read_b128`
- `MFMA` = 2 kp × 4 rows × 8 cols = 64 条 `v_mfma_f32_16x16x32_bf16`

## 四、pass 的三个阶段

### 阶段 1：从输入 IR 中提取三个组件

pass 做的第一件事是把 k-loop body 拆成三部分：

```python
# 1. 提取 G2S stores
g2s_a, g2s_b, remaining = _find_async_blocks(body_stmts)
# g2s_a = [vec_store_0, vec_store_1, vec_store_2, vec_store_3]  ← 4 条 A 的 G2S
# g2s_b = [vec_store_0, vec_store_1, vec_store_2, vec_store_3]  ← 4 条 B 的 G2S

# 2. 提取 consumer (async_wait 后面的 S2R + MFMA)
consumer_body, _found = _find_consumer(remaining)

# 3. 从 consumer 中提取 MFMA stages
mfma_stages, alloc_wrappers = _extract_consumer_stages(consumer_body)
```

#### `_find_async_blocks` 的工作方式

扫描 loop body 顶层语句，找到 `async_scope` / `async_commit_queue_scope` 属性的语句。
第一个是 A 的 G2S，第二个是 B 的 G2S。对每个 async block，展开 unroll 循环，得到
独立的 vectorized store 列表。

#### `_find_consumer` 的工作方式

在剩余语句中找 `async_wait_queue_scope` 属性的语句，返回 wait 内部的 body（即 S2R + MFMA）。

#### `_extract_consumer_stages` 的工作方式

consumer body 包含：`S2R_A_loop`, `S2R_B_loop`, `MFMA_loop`。
由于原始 IR 中这些是普通 For 循环（没有 `sched_barrier` 标记），
pass 调用 `_create_stages_from_raw_consumer` 来自动创建 4 个 stage。

### 阶段 2：把 consumer 拆成 4 个 stage (`_create_stages_from_raw_consumer`)

这是最复杂的部分。目标是把 S2R + MFMA 拆成 4 个均匀的阶段。

**拆分维度**：

```
MFMA loop: for kp in range(2):       ← k_pack 维度 (bf16 需要 2 次 16x16x32)
              for i in range(4):      ← warp_rows (M tile 内的行)
                for j in range(8):    ← warp_cols (N tile 内的列)
                  mfma(B[j*2+kp], A[i*2+kp], C[i*8+j])
```

按 kp 和 j 的上下半拆分：

| Stage | kp | j 范围 | S2R loads | MFMA 数量 |
|-------|-----|---------|-----------|-----------|
| 0 | 0 | 0..3 (lo) | A(kp=0) 4条 + B(kp=0,j_lo) 4条 | 4×4=16 |
| 1 | 0 | 4..7 (hi) | B(kp=0,j_hi) 4条 | 4×4=16 |
| 2 | 1 | 0..3 (lo) | A(kp=1) 4条 + B(kp=1,j_lo) 4条 | 4×4=16 |
| 3 | 1 | 4..7 (hi) | B(kp=1,j_hi) 4条 | 4×4=16 |

每个 stage 的内部格式：

```
sched_barrier(0)      ← 防止 LLVM 跨越此点重排
S2R loads...           ← ds_read 指令
sched_barrier(0)      ← 防止 LLVM 把 G2S/S2R 和 MFMA 混排
setprio(1)            ← 提高 MFMA 优先级
16× MFMA              ← v_mfma 指令
setprio(0)            ← 恢复默认优先级
```

**S2R 拆分的关键**：

S2R A 的原始循环是 `for i_1, local_id in grid(4, 2)`，展开后有 8 条 ds_read。
`local_id` 维度对应 k_pack：
- `local_id=0` → kp=0 的数据
- `local_id=1` → kp=1 的数据

展开顺序是 `(i_1=0,lid=0), (i_1=0,lid=1), (i_1=1,lid=0), ...`，
所以 even index = kp=0，odd index = kp=1。

S2R B 的原始循环是 `for j, local_id in grid(8, 2)`，同理按 local_id 分 kp，
再按 j 的上下半分。

### 阶段 3：把 G2S stores 穿插到 4 个 stage 中 (`_interleave_one_stage`)

8 条 G2S（4A + 4B）均匀分配到 4 个 stage，每个 stage 2 条。
G2S 插入位置：在 S2R loads 之后、第二个 `sched_barrier` 之前。

```
每个 stage 最终结构:

sched_barrier(0)
S2R loads...          ← ds_read (从当前 k 的 shared memory 读)
G2S loads (2条)       ← buffer_load_b128...lds (发 k+1 的 global→shared)
sched_barrier(0)
[vmcnt(0)]            ← 仅 last stage：等待所有 G2S 完成
[s_barrier]           ← 仅 last stage：workgroup 同步
setprio(1)
16× MFMA
setprio(0)
```

**G2S 的包装方式**：

```python
def _make_raw_g2s(inner_store_loop):
    return AttrStmt(0, "async_scope", 1, inner_store_loop)
```

只用 `async_scope`，**不用** `async_commit_queue_scope`。
这是因为后续的 `InjectPTXAsyncCopy` pass 看到 `async_scope` 内的
shared←global BufferStore 会自动转成 `ptx_cp_async`（即 `buffer_load_b128...lds`）。
如果加了 `async_commit_queue_scope`，`ThreadSync` pass 会在相邻 commit 之间
插入 `tvm_storage_sync`，翻译成额外的 barrier。

## 五、输出 IR 结构

```
prologue:
  G2S A[0] (async_scope, 不变)
  G2S B[0] (async_commit, 不变)
  vmcnt(0) + s_barrier               ← 新增：等待 prologue G2S 完成

k-loop: for k in range(127):         ← 范围不变
  ┌─ Stage 0 ─────────────────────────────────────────┐
  │ sched_barrier(0)                                   │
  │ S2R: A_local[0,16,32,48]  (A kp=0, 4 ds_reads)   │
  │      B_local[0,16,32,48]  (B kp=0 j_lo, 4 reads) │
  │ G2S: A[0], A[1]  (2 buffer_load_b128...lds)       │
  │ sched_barrier(0)                                   │
  │ setprio(1)                                         │
  │ 16× MFMA (kp=0, i=0..3, j=0..3)                  │
  │ setprio(0)                                         │
  ├─ Stage 1 ─────────────────────────────────────────┤
  │ sched_barrier(0)                                   │
  │ S2R: B_local[64,80,96,112] (B kp=0 j_hi, 4 reads)│
  │ G2S: A[2], A[3]                                    │
  │ sched_barrier(0)                                   │
  │ setprio(1)                                         │
  │ 16× MFMA (kp=0, i=0..3, j=4..7)                  │
  │ setprio(0)                                         │
  ├─ Stage 2 ─────────────────────────────────────────┤
  │ sched_barrier(0)                                   │
  │ S2R: A_local[8,24,40,56]  (A kp=1, 4 ds_reads)   │
  │      B_local[8,24,40,56]  (B kp=1 j_lo, 4 reads) │
  │ G2S: B[0], B[1]                                    │
  │ sched_barrier(0)                                   │
  │ setprio(1)                                         │
  │ 16× MFMA (kp=1, i=0..3, j=0..3)                  │
  │ setprio(0)                                         │
  ├─ Stage 3 ─────────────────────────────────────────┤
  │ sched_barrier(0)                                   │
  │ S2R: B_local[72,88,104,120] (B kp=1 j_hi, 4 reads)│
  │ G2S: B[2], B[3]                                    │
  │ sched_barrier(0)                                   │
  │ vmcnt(0)    ← 等待 8 条 G2S 全部完成               │
  │ s_barrier   ← workgroup 同步（所有线程的 G2S 可见） │
  │ setprio(1)                                         │
  │ 16× MFMA (kp=1, i=0..3, j=4..7)                  │
  │ setprio(0)                                         │
  └────────────────────────────────────────────────────┘

epilogue: (不变)
  async_wait(0)
  S2R + MFMA for last tile
```

## 六、调用链

```
InjectCdna4Pipeline()              ← TVM pass 入口
  └─ InjectCdna4PipelineMutator
       └─ visit_for_()            ← 找到 k-loop（serial, extent>1, 含 async_scope）
            └─ _try_cdna4_pipeline(loop)
                 ├─ _find_async_blocks()           ← 提取 G2S A/B stores
                 ├─ _find_consumer()               ← 提取 consumer (S2R + MFMA)
                 ├─ _extract_consumer_stages()     ← 拆成 4 个 stage
                 │    └─ _create_stages_from_raw_consumer()  ← 自动拆分
                 │         ├─ _unroll_serial_fors()  ← 展开 S2R / MFMA 循环
                 │         ├─ 按 kp + j_half 分组
                 │         └─ _build_stage()         ← 组装 stage 格式
                 └─ _interleave_one_stage() × 4    ← 把 G2S 插入每个 stage
```

## 七、依赖的 helper 函数（来自 interleave_g2s.py）

| 函数 | 作用 |
|------|------|
| `_flatten_seq(stmt)` | 把嵌套的 SeqStmt 展平成列表 |
| `_find_async_blocks(stmts)` | 从 loop body 中分离出 G2S async blocks |
| `_extract_g2s_from_async_block(attr)` | 展开 unrolled G2S 循环为独立的 vec stores |
| `_find_consumer(stmts)` | 找到 async_wait 包裹的 consumer body |
| `_extract_consumer_stages(body)` | 从 consumer 中提取 MFMA stages |
| `_create_stages_from_raw_consumer(body)` | 从原始 S2R+MFMA 循环自动创建 4 stage |
| `_unwrap_alloc_decl(body)` | 剥离 DeclBuffer/Allocate/AttrStmt 包装 |
| `_rewrap(body, wrappers)` | 重新包装 |
| `_contains_mfma(node)` | 递归检查是否含 MFMA 调用 |
| `_is_s2r_loop(node)` | 判断是否为 S2R（shared→local）循环 |
| `_unroll_serial_fors(node)` | 展开所有 serial For，保留 vectorized |

## 八、当前已知问题

**正确性错误（24% mismatch, NaN）**：

pass 生成的 IR 在最终执行时产生错误结果。已排除的原因：
- ~~`tvm_storage_sync` 干扰~~：`RemoveRedundantSync` pass 已清理循环内的 sync
- ~~`__syncthreads` 隐含 vmcnt(0)~~：HIP 上 `tvm_storage_sync` 翻译为 `s_barrier`，不影响 vmcnt

**待排查方向**：
1. S2R 的 kp 拆分是否正确（`local_id=0` 真的对应 kp=0 吗？）
2. MFMA 的 j-half 拆分是否和 S2R 的 j-half 对应
3. prologue 中缺少对初始 G2S 的 wait（目前 prologue vmcnt(0) 应该已经覆盖）
4. epilogue 部分是否受影响
