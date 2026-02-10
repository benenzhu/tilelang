# G2S Copy: Swizzle 从 Store 侧交换到 Load 侧

## 1. 目标

在 AMD ROCm (gfx950+) 上，让 Global → Shared Memory 的 copy 的 **LDS 写入地址连续**（按 `lane_id * N` 排列），为后续使用硬件 `buffer_load_b128 ... lds` 指令做准备。

**核心思路**：当前 swizzle（XOR）被施加在 shared memory 的 store 侧（左边），导致每个线程写入 LDS 的地址是非连续的。我们要把这个 swizzle **交换到 global memory 的 load 侧（右边）**，让每个线程改为从 global memory 的不同位置读取，但写入 LDS 的地址变为连续。

最终 LDS 里的内容（swizzle 布局）保持不变——只是不同线程读取了不同的 global 元素。

## 2. 等价性证明

设 swizzle 函数为 `S(row, col)`，它是一个基于 XOR 的列索引置换（对每个固定的 row，S 是 col 的双射，且 XOR 是自逆的）。

**交换前（当前行为）**：
```
LDS[S(row, col)] = Global[base + (row, col)]
```
- 线程 T 负责 `(row, col)`，写到 LDS 的 swizzled 地址。

**交换后（目标行为）**：
```
LDS[(row, col)] = Global[base + (row, S(row, col))]
```
- 线程 T 仍负责 `(row, col)`，但写到 LDS 的 **顺序地址**，从 global 的 **swizzled 位置** 读取。

验证：对于任意物理位置 P = S(row, col)：
- 交换前：`LDS[P] = Global[base + (row, S⁻¹(row, P))]`
- 交换后：令 `col' = S⁻¹(row, P) = S(row, P)`（XOR 自逆），线程 `(row, col')` 写入 `LDS[col'] = Global[base + (row, S(row, col'))] = Global[base + (row, P)]`
- 等价于交换前 `LDS[P] = Global[base + (row, S⁻¹(row, P))]`。✓

Global memory 的 coalescing 也不受影响——同一 warp 内的线程仍然访问连续（或接近连续）的 global 地址，只是排列顺序略有变化。

## 3. 修改位置

修改位于 **`src/transform/lower_tile_op.cc`** 的 `LowerTileOpPass::VisitStmt_(const BufferStoreNode *op)` 方法。

这是 `LowerTileOp` pass 处理 `BufferStore` 节点的地方。当检测到：
1. 目标是 ROCm
2. store 的目的 buffer 是 shared memory
3. store 的 value 是从 global memory 的 `BufferLoad`（可能包裹在 `Cast` 或 `if_then_else` 中）

就执行 swizzle 交换。

### 当前代码（已写入但条件检查待修复）

```cpp
// lower_tile_op.cc, VisitStmt_(BufferStoreNode*) 中
if (buffer_remap_.count(buffer)) {
    auto layout = layout_map_[buffer];
    auto new_buffer = buffer_remap_[store->buffer];
    
    if (TargetIsRocm(target_) && !is_ptx_ && IsSharedBuffer(buffer)) {
        // 提取 global BufferLoad ...
        // 匹配模式: BufferLoad / Cast(BufferLoad) / if_then_else(pred, BufferLoad, zero)
        
        if (load_node && /* 维度检查 */) {
            auto swizzled = layout->Forward(store->indices);
            // delta[k] = swizzled[k] - store->indices[k]
            // new_load[k] = load->indices[k] + delta[k]
            // return BufferStore(new_buffer, new_value, store->indices);  // 未 swizzle 的顺序地址
        }
    }
    // 默认路径: return BufferStore(new_buffer, ..., layout->Forward(store->indices));
}
```

## 4. 当前卡点

### 问题：layout 改变了维度结构（reshape）

从 pass dump 可以看到：

**pass_04 (LayoutInference 后, LowerTileOp 前)**:
```python
A_shared = T.Buffer((256, 64), "bfloat16", scope="shared.dyn")  # 2D
```

**pass_05 (LowerTileOp 后)**:
```python
A_shared = T.Buffer((1, 32, 512), "bfloat16", scope="shared.dyn")  # 3D!
```

layout 不仅做了 XOR swizzle，还将维度 reshape 了：
```
[row, col]  →  [row // 8, row % 8 * 64 + swizzle(col)]
  (256, 64)       (32,           512)
```

加上 replication 前缀维度 `1`，最终 buffer 是 3D `[1, 32, 512]`。

### "delta 法" 的局限

当前的 **delta 法** 假设 layout 保持维度结构不变（如 2D → 2D，每个维度独立变化）：
```
delta[k] = layout.Forward(indices)[k] - indices[k]
```

但当 layout 做了 reshape（如 `[256, 64] → [32, 512]`），delta 不再有意义：
- `delta[0] = row//8 - row` — 巨大的负数，不可用
- `delta[1] = (row%8*64 + swizzle_col) - col` — 混入了 row 的信息，不能简单加到 global load 的 col 上

### `Layout::Forward` 的 prefix 机制

`Layout::Forward` 支持 `vars.size() > InputDim()`，会将前缀 indices 原样传递：
```cpp
// Forward([stage, row, col]) with InputDim=2:
// → [stage, transform(row, col)[0], transform(row, col)[1]]
```

这意味着 `store->indices` 在 LowerTileOp 时可能已经是 3D（含 stage 前缀），而 `load->indices` 仍是 2D（global buffer 没有 stage 维度），导致 `store->indices.size() != load->indices.size()` 条件判断失败。

### 维度检查导致 swap 被跳过

当前添加的安全检查：
```cpp
if (load_node &&
    store->indices.size() == load_node->indices.size() &&   // 可能 3 != 2
    layout->InputDim() == layout->OutputDim() &&             // 可能 2 == 2 ✓
    layout->OutputDim() == new_buffer->shape.size())         // 可能 2 != 3
```

这些检查因为维度不匹配而跳过了 swap，回退到默认的 swizzle-on-store 路径。

## 5. 需要解决的问题

### 核心难题
layout 同时做了 **reshape** 和 **swizzle**，这两个操作耦合在一起。我们需要的是：
- **Store 侧**：只做 reshape（不做 XOR swizzle）→ 地址连续
- **Load 侧**：将 XOR swizzle 施加到 global 读取索引上

但当前没有简单的方式将 layout 分解为 `reshape ⊕ swizzle`。

### 可能的解决方案

#### 方案 A：在 1D（flatten）空间做 delta
将 swizzled 和 unswizzled 的物理地址都展平为 1D offset：
```
flat_swizzled   = row * 64 + swizzle(col)
flat_sequential = row * 64 + col
delta_flat = swizzle(col) - col
```
delta 只影响 col，不影响 row。对 global load 的 col 维度加上 `swizzle(col) - col` 即可。

**挑战**：需要从 layout 的 Forward 表达式中提取出纯 swizzle 部分（即 col 维度的变化），而不是整体的 reshape + swizzle。

#### 方案 B：直接构造 swizzle 表达式
不依赖 layout 的 Forward，而是在检测到 `makeMatrixCoreSwizzleLayout` 类型的 layout 时，直接构造 XOR swizzle 函数：
```
swizzle_col(row, col) = ((col / vecSize) ^ (row % maxPhase)) * vecSize + col % vecSize
```
然后：
- Store 侧：使用 `layout_without_swizzle->Forward(indices)` 或等价的 reshape-only 变换
- Load 侧：将 col 替换为 `swizzle_col(row, col)`

**挑战**：需要从 layout 中识别出它是 `makeMatrixCoreSwizzleLayout` 并提取参数 `vecSize` 和 `maxPhase`。

#### 方案 C：修改 layout 基础设施
在 Layout 类中添加 `ForwardWithoutSwizzle()` 方法，或将 layout 分解为 `reshape_layout + swizzle_permutation`。

**挑战**：改动范围大，影响面广。

#### 方案 D：在 Copy 生成阶段（而非 LowerTileOp）交换
在 `CopyNode::LowerNormalCopy`（`src/op/copy.cc`）中，当 target 是 ROCm 且 copy 是 G2S 时，在生成 SIMT loop **之前**就交换 swizzle。此时 indices 还在逻辑 2D 空间，没有 reshape。

**具体做法**：在 `MakeSIMTLoop` 生成 `BufferStore(shared, BufferLoad(global, src_indices), dst_indices)` 时，将 `src_indices` 的 col 维度替换为 swizzle_col，并标记该 copy 不需要在 LowerTileOp 中再次 swizzle store 侧。

**挑战**：需要在 copy lowering 阶段获取 swizzle layout 信息，并协调 LowerTileOp 不对这个 buffer 做 store 侧 swizzle。

## 6. 相关文件

| 文件 | 作用 |
|------|------|
| `src/transform/lower_tile_op.cc` | 当前修改位置。`VisitStmt_(BufferStoreNode*)` 应用 layout 变换 |
| `src/layout/gemm_layouts.cc` | `makeMatrixCoreSwizzleLayout` — XOR swizzle layout 定义 |
| `src/layout/layout.cc` | `Layout::Forward`, `Layout::OutputShape`, `Layout::Inverse` |
| `src/op/copy.cc` | Copy 操作的 lowering（`LowerNormalCopy`, `MakeSIMTLoop`） |
| `src/op/parallel.cc` | `IfBufferRemapLoopGenerator` — 在某些路径中应用 layout |
| `src/transform/inject_ptx_async_copy.cc` | 将 `BufferStore(shared, BufferLoad(global))` 转为 `ptx_cp_async` |
| `src/tl_templates/hip/copy.h` | `tl::cp_async_gs` 模板定义（同步 VGPR 路径 vs 硬件 async 路径）|
| `src/transform/loop_partition.cc` | `LowerParallelLoop` — 循环分块和向量化 |

## 7. 当前代码状态

已修改文件：`src/transform/lower_tile_op.cc`
- 在 `VisitStmt_(BufferStoreNode*)` 中添加了 G2S swizzle 交换逻辑
- 但由于维度检查（layout reshape 导致维度不匹配），swap 被跳过
- 默认路径有调试用的 `LOG(INFO)` 语句（应在最终版本中删除）

下一步需要选择上述方案之一来解决 reshape + swizzle 耦合的问题。
