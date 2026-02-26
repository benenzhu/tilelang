 Step A: Python creates the base layout with `input_size = [256, 64]`


```python
  # swizzle.py:66-77
  def make_swizzled_layout(buffer, ...):
      stride, continuous = 256, 64          # from buffer shape
      base = _ffi_api.make_swizzled_layout(256, 64, element_size, ...)
      return base.reshape(shape)            # reshape([256, 64]) = identity → no-op
```
```c++
  The FFI calls MakeFullBankSwizzleLayout2D(256, 64, element_size) in gemm_layouts.cc:485-501:
  static Layout MakeFullBankSwizzleLayout2D(int stride=256, int continuous=64, int element_size=16) {
      int vector_size = 128 / 16 = 8;     // 8 elements per 128-bit vector
      PrimExpr ts = FloorDiv(_i, 8);       // _i / 8  → range [0, 32)
      PrimExpr s  = FloorMod(_i, 8);       // _i % 8  → range [0, 8)
      PrimExpr tc = FloorDiv(FloorDiv(_j, 8), 8);  // (_j/8)/8 → range [0, 1) = 0
      PrimExpr c  = FloorMod(FloorDiv(_j, 8), 8);  // (_j/8)%8 → range [0, 1)
      PrimExpr vec = FloorMod(_j, 8);      // range [0, 8)
      PrimExpr c_swizzle = xor8x8(c, s);   // XOR swizzle
      PrimExpr index = vec + (c_swizzle + s * 8) * 8;
      //                                       ↑ range [0, 512)
      return Layout({256, 64}, {tc, ts, index});
      //            input_size   forward_index (3 outputs)
  }
```

then: 
```c++
  OutputShape() analyzes each forward_index_[k] to compute its range:
  - tc = (_j/8)/8 with _j ∈ [0, 64) → tc ∈ [0, 1) → output_shape[0] = 1
  - ts = _i/8 with _i ∈ [0, 256) → ts ∈ [0, 32) → output_shape[1] = 32
  - index = complex expression → range [0, 512) → output_shape[2] = 512
  So OutputShape() = [1, 32, 512].
```

```c++
Step C: makeBufferWithLayout in lower_tile_op.cc:34-82 creates the physical buffer

Array<PrimExpr> layout_shape = layout->OutputShape();   // [1, 32, 512]
// For shared memory, compute replicate_extent:
buffer_extent = 256 * 64 = 16384
layout_extent = 1 * 32 * 512 = 16384
replicate_extent = 16384 / 16384 = 1   // no replication needed
// output_shape = [1, 32, 512]  (or simplified to [32, 512] if tc is always 0)

```




```c++
  Question 3: How is swizzle applied in g2s and s2r?

  G2S (global → shared): VisitStmt_(BufferStoreNode) at line 992-993

  // Original IR: A_shared[row, col] = A_global[global_row, global_col]
  auto new_indices = layout_map_[buffer]->Forward(store->indices);
  return BufferStore(new_buffer, store->value, new_indices);

  Forward([row, col]) substitutes _i=row, _j=col into the forward_index_ expressions, producing [tc, ts, swizzled_index]. The store becomes:
  A_shared_physical[tc, ts, swizzled_index] = A_global[global_row, global_col]

  The global load is unchanged. The swizzle only affects where in shared memory the data lands.

  S2R (shared → register): Two paths

  Path A: Scalar loads — VisitExpr_(BufferLoadNode) at line 869

  auto new_indices = layout_map_[buffer]->Forward(load->indices);
  return BufferLoad(new_buffer, new_indices);

  Same Forward applied to load indices. The consumer reads from the swizzled address, which matches where g2s stored the data.
```


```c++
  Correctness: This works because XOR swizzle is an involution (self-inverse). The original semantics shared[swizzle(pos)] = global[pos] becomes
  shared[pos] = global[pos + delta]. When the consumer later reads shared[swizzle(pos)], it gets global[swizzle(pos) + delta'] — and since the same
  swizzle formula applies to the read, the right data lands at the right place.
```