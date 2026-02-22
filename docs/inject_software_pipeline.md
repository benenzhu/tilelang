# InjectSoftwarePipeline 详解

## 整体流程

TileLang 的软件流水线分两个 pass：

```
T.Pipelined(num_stages=2):
    T.copy(A_g, A_shared)     # stage 0 (G2S)
    T.copy(B_g, B_shared)     # stage 0 (G2S)
    T.gemm(A_shared, B_shared, C)  # stage 1 (compute)
        |
        v
PipelinePlanning (pipeline_planning.cc)
    -> 给每条语句标注 stage 和 order
        |
        v
InjectSoftwarePipeline (inject_pipeline.cc)
    -> 把循环变成 prologue + loop body + epilogue
```

## PipelinePlanning 做什么

**文件**: `src/transform/pipeline_planning.cc` (738行)

### 1. 识别 producer vs consumer

`BufferRegionCollector` 分析每条语句：
- **读 global + 写 shared** → `copy_stage = true` (G2S 操作，标记为 stage 0)
- **其他** → compute stage (标记为 stage = num_stages - 1)

### 2. 输出标注

给 for 循环加上三个 annotation：
```
software_pipeline_stage = [0, 0, 1]   # 每条语句的 stage
software_pipeline_order = [0, 1, 2]   # 执行顺序
software_pipeline_async_stages = [0]  # 哪些 stage 是 async 的
```

## InjectSoftwarePipeline 做什么

**文件**: `src/transform/inject_pipeline.cc` (1415行)

### 核心概念

对于 `num_stages=2` 的 pipeline，loop `for k in [0, N)`：

```
原始:
  for k = 0..N-1:
    G2S(k)      # stage 0
    compute(k)  # stage 1

变换后:
  prologue:  k=0 时只执行 stage 0 → G2S(0)
  body:      k=1..N-1 时执行 stage 0 的 G2S(k) + stage 1 的 compute(k-1)
  epilogue:  k=N 时只执行 stage 1 → compute(N-1)
```

### 关键步骤

#### Step 1: Buffer 多版本化 (Double Buffering)

```cpp
num_versions = use_stage - def_stage + 1
// stage 0 写, stage 1 读 → num_versions = 1 - 0 + 1 = 2
```

`A_shared[M, K]` → `A_shared[2, M, K]`

所有 buffer 访问加上版本索引：
```cpp
A_shared[i, j] → A_shared[floormod(k, 2), i, j]   // k%2 选择 buffer slot
```

#### Step 2: Loop variable skewing

每个 stage 使用不同的 k 值：
```
stage 0 (G2S):   skewed_k = k           // 当前 k
stage 1 (compute): skewed_k = k - 1     // 上一个 k
```

这就是为什么 loop body 里 G2S 加载 k+1 的数据，compute 使用 k 的数据。

#### Step 3: Emit prologue / body / epilogue

```cpp
// 用 EmitImpl 生成三段代码
Stmt prologue = EmitImpl(0, max_stage, unroll=true);     // k=0 只有 G2S
Stmt body     = EmitImpl(max_stage, extent, unroll=false); // G2S(k) + compute(k-1)
Stmt epilogue = EmitImpl(extent, extent+max_stage, unroll=true); // 只有 compute
```

`EmitImpl` 对每个 k 值：
1. 对每条语句计算 `skewed_k = k - stage`
2. 检查 `skewed_k` 是否在 `[0, N)` 范围内
3. 用 `skewed_k` 替换原来的 loop variable
4. 用 `floormod(skewed_k, num_versions)` 替换 buffer 索引

#### Step 4: Async 标记

```
async_scope → 标记异步操作 (变成 cp_async / buffer_load_lds)
async_commit_queue_scope → commit group 边界
async_wait_queue_scope + async_wait_inflight_count → wait point
```

嵌套结构：
```
AttrStmt("async_commit_queue_scope", 0,
  AttrStmt("async_scope", 1,
    BufferStore(A_shared, ..., BufferLoad(A_global, ...))  // G2S
  )
)
// ...
AttrStmt("async_wait_queue_scope", 0,
  AttrStmt("async_wait_inflight_count", 1,  // 1 个 group 可以 in-flight
    compute_body  // S2R + MFMA
  )
)
```

`async_wait_inflight_count = 1` 表示等到只剩 1 个 commit group 还在 flight：
- 有 2 个 commit group（prologue 的 + body 当前的）
- wait(1) 意味着 prologue 的必须完成，当前的可以继续

## 生成的 TIR 结构

```python
# Prologue: 加载 k=0 数据
with async_scope:
    for i in unroll(4):
        A_shared[0, ...] = A[...]  # k=0 数据到 slot 0

with async_commit_queue_scope:
    with async_scope:
        for i in unroll(4):
            B_shared[0, ...] = B[...]

# Main loop
for k in range(127):  # k=0..126
    # G2S for k+1 (stage 0, skewed_k = k)
    with async_scope:
        for i in unroll(4):
            A_shared[(k+1)%2, ...] = A[..., k+1 offset]

    with async_commit_queue_scope:
        with async_scope:
            for i in unroll(4):
                B_shared[(k+1)%2, ...] = B[..., k+1 offset]

    # Wait + Compute for k (stage 1, skewed_k = k-1+1 = k)
    async_wait(inflight=1):
        for ki in range(2):
            S2R + MFMA(A_shared[k%2], B_shared[k%2])

# Epilogue: 最后一个 k 的 compute
async_wait(inflight=0):
    for ki in range(2):
        S2R + MFMA(A_shared[last_k%2], B_shared[last_k%2])

# Store C
for i in unroll(32):
    C[...] = C_local_cast[...]
```

## 如何修改来实现 2-step-ahead schedule

你要的 schedule 和当前 pipeline 的区别：

| | 当前 (num_stages=2) | 你要的 (2-step-ahead) |
|---|---|---|
| Prologue | 加载 k=0 全部 (8 subtile) | 加载 k=0 全部 + k=1 部分 (7 subtile) |
| G2S/iter | 8 个，全部去 `(k+1)%2` | 8 个，2个去 `(k+1)%2`，6个去 `k%2` |
| vmcnt | wait(1) → vmcnt(8) → vmcnt(0) | vmcnt(6) |
| Pipeline depth | 1 k-step | 2 k-steps |

### 方案 A: 修改 InjectSoftwarePipeline

在 `inject_pipeline.cc` 的 `EmitImpl` 中：

1. **加一个 `pipeline_mode` 参数**（normal vs two_step_ahead）
2. 在 two_step_ahead 模式下：
   - Prologue 多发一轮 stage 0（加载 k=1 的部分数据）
   - Body 里的 stage 0 语句被拆分：部分用 `skewed_k = k`（k+1 数据），部分用 `skewed_k = k+1`（k+2 数据）
   - `async_wait_inflight_count` 改成 6（而不是 1）

**难点**: `inject_pipeline.cc` 是 C++ 代码，修改需要重新编译。而且 `EmitImpl` 的逻辑是通用的，加特殊逻辑会破坏通用性。

### 方案 B: 新 pass 替换 InjectSoftwarePipeline (更推荐)

写一个 Python pass `InjectHipKittensPipeline`：

1. 识别 `num_stages=2` + HIP target 的情况
2. 不调用 `InjectSoftwarePipeline`，改用自己的 pipeline injection
3. 完全控制 prologue、loop body、epilogue 的结构
4. 直接 emit `tir.call_extern("s_waitcnt vmcnt(6)")` 绕过 async 机制

在 `phase.py` 中：
```python
if _is_hip_target(target) and _should_use_hipkittens_pipeline(pass_ctx):
    mod = tilelang.transform.InjectHipKittensPipeline()(mod)
else:
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
```

### 方案 C: 在 _gemm_ssr 中内联 pipeline (最简单)

不用 `T.Pipelined`，直接在 `_gemm_ssr()` 里手写 prologue + loop + epilogue：

```python
@T.prim_func
def _gemm_with_pipeline():
    # Prologue: 7 G2S loads
    T.copy(A_g[..., 0], A_shared[0, 0])  # As[0][0]
    T.copy(A_g[..., 0], A_shared[0, 1])  # As[0][1]
    ...
    vmcnt(6)

    for k in T.serial(0, K // block_K):
        # Phase 0: S2R + G2S(As[(k+1)&1][1]) + MFMA
        # Phase 1: S2R + G2S(As[k&1][0]) + MFMA
        # Phase 2: S2R + G2S(Bs[k&1][0]) + MFMA
        # Phase 3: S2R + G2S(Bs[k&1][1]) + vmcnt(6) + MFMA

    # Epilogue
    vmcnt(0)
    # last S2R + MFMA
```

**难点**: `_gemm_ssr()` 目前不接触 G2S，需要传入 global buffer 引用。需要改 `T.gemm` 的接口。

## 建议

**方案 B（新 Python pass）最实际**：
- 不用改 C++ 代码
- 完全控制 pipeline 结构
- 可以用 `tir.call_extern` 直接 emit vmcnt
- 只对 HIP target 生效，不影响其他 target

操作在 `InjectSoftwarePipeline` 之前。检测到 `T.Pipelined` + HIP，就用自己的 pass 处理 pipeline injection，跳过默认的 `InjectSoftwarePipeline`。
