/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Replace copy from global to shared with async copy
 * \file inject_ptx_async_copy.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

static bool TargetIsRocm(const Target &target) {
  return target->kind->name == "rocm" || target->kind->name == "hip";
}

// Check if a flat shared-memory offset expression is truly LINEAR
// (affine) in the thread variable — i.e. f(tx) = a*tx + b.
//
// buffer_load_lds writes to m0 + lane_id * N, requiring strict linearity.
// A 2D tiling pattern like ((tx & 31) >> 3) * 2048 + (tx & 7) * 8 is NOT
// linear even though f(1)-f(0) is constant.
//
// We check f(k+1)-f(k) == f(1)-f(0) at multiple sample points to catch
// non-linear patterns (floor-div, bit-extract, modular arithmetic).
static bool IsLdsContiguous(const PrimExpr &flat_offset) {
  arith::Analyzer analyzer;
  std::unordered_set<const VarNode *> seen;
  Array<Var> free_vars;
  PostOrderVisit(flat_offset, [&](const ObjectRef &node) {
    if (auto *v = node.as<VarNode>()) {
      if (seen.insert(v).second) {
        free_vars.push_back(Downcast<Var>(node));
      }
    }
  });
  for (const auto &var : free_vars) {
    auto name = std::string(var->name_hint);
    if (name == "tx" || name == "thread_binding" ||
        name.find("thread") != std::string::npos) {
      // Check stride at multiple points to detect non-linearity.
      // Sample at 0, 1, 7, 8, 31, 32, 63 to catch common tile boundaries.
      static const int sample_points[] = {0, 1, 7, 8, 31, 32, 63};
      PrimExpr f0 = analyzer.Simplify(Substitute(
          flat_offset, Map<Var, PrimExpr>{{var, IntImm(var->dtype, 0)}}));
      PrimExpr f1 = analyzer.Simplify(Substitute(
          flat_offset, Map<Var, PrimExpr>{{var, IntImm(var->dtype, 1)}}));
      PrimExpr expected_stride = analyzer.Simplify(f1 - f0);
      auto *stride_imm = expected_stride.as<IntImmNode>();
      if (!stride_imm) return false;

      for (int pt : sample_points) {
        if (pt == 0) continue;
        PrimExpr fk = analyzer.Simplify(Substitute(
            flat_offset, Map<Var, PrimExpr>{{var, IntImm(var->dtype, pt)}}));
        PrimExpr actual = analyzer.Simplify(fk - f0);
        PrimExpr expected = IntImm(DataType::Int(32), stride_imm->value * pt);
        if (!analyzer.CanProveEqual(actual, expected)) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

class PTXAsyncCopyInjector : public StmtMutator {
public:
  explicit PTXAsyncCopyInjector(bool is_rocm) : is_rocm_(is_rocm) {}
  Stmt VisitStmt_(const AttrStmtNode *attr) {
    if (attr->attr_key == tir::attr::async_scope) {
      ICHECK(in_async == false) << "Nested async scopes not supported";
      in_async = true;
      auto body = this->VisitStmt(attr->body);
      in_async = false;
      return body;
    }
    return StmtMutator::VisitStmt_(attr);
  }

  Stmt InjectPTX(const BufferLoadNode *load, const BufferStoreNode *store,
                 bool predicated = false,
                 const PrimExpr &predicate_value = PrimExpr()) {
    if (load->buffer.scope() == "global") {
      ICHECK(load->indices.size() == 1 && store->indices.size() == 1);
      ICHECK(load->indices[0]->dtype.lanes() ==
             store->indices[0]->dtype.lanes())
          << load->indices[0] << " vs. " << store->indices[0] << " with lanes "
          << load->indices[0]->dtype.lanes() << " vs. "
          << store->indices[0]->dtype.lanes();

      const int indices_lanes = load->indices[0]->dtype.lanes();
      const int bytes = indices_lanes * load->buffer->dtype.bytes();

      if (bytes == 4 || bytes == 8 || bytes == 16) {
        auto dst_elem_type =
            GetPointerType(store->buffer->data->type_annotation);
        auto src_elem_type =
            GetPointerType(load->buffer->data->type_annotation);
        ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
            << "Both store and load buffer should have a pointer type "
               "annotation.";

        int index_factor = 1;
        if (dst_elem_type.value() != src_elem_type.value()) {
          // The only case where src and dst have different dtypes is when the
          // dst shared memory is a byte buffer generated by merging dynamic
          // shared memory.
          ICHECK(store->buffer.scope() == "shared.dyn" ||
                 store->buffer.scope() == "shared");
          ICHECK(dst_elem_type.value() == DataType::UInt(8));
          // BufferStore/Load have the "pointer reinterpret" semantics according
          // to their "value" dtype. Their "indices" are supposed to be applied
          // after such pointer cast, for example:
          // ((*float16)(byte_buffer))[buffer->indices] = fp16_value; To replace
          // BufferStore/Load with cp.async, we need to multiply the store index
          // by the byte size of the "value" dtype, to get the correct offset
          // into the byte buffer.
          index_factor = src_elem_type->bytes();
        }

        if (indices_lanes == 1) {
          auto src_offset = load->indices[0];
          auto dst_offset = store->indices[0];

          // Calculate the number of elements based on bytes and dtype
          int dst_elem_count = bytes / dst_elem_type->bytes();
          int src_elem_count = bytes / src_elem_type->bytes();

          // Create access_ptr for destination (shared memory, write access)
          auto dst_access_ptr = store->buffer.access_ptr(
              2, DataType::Handle(), 1, dst_offset, PrimExpr(dst_elem_count));

          // Create access_ptr for source (global memory, read access)
          auto src_access_ptr = load->buffer.access_ptr(
              1, DataType::Handle(), 1, src_offset, PrimExpr(src_elem_count));

          ffi::Array<PrimExpr> cp_async_args;
          if (predicated) {
            cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes),
                             predicate_value};
          } else {
            cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes)};
          }
          // On ROCm, use buffer_load...lds when LDS is lane-contiguous.
          bool lds_contiguous = is_rocm_ && !predicated && bytes == 16 &&
                                IsLdsContiguous(dst_offset);
          const Op &op = lds_contiguous
                             ? tl::ptx_cp_async_lds()
                             : tvm::tir::builtin::ptx_cp_async();
          return Evaluate(Call(store->buffer->dtype, op, cp_async_args));
        }

        // Predicated load don't support vectorized indexing.
        if (!predicated) {
          // Only some vectorized indexing patterns are supported for now.
          auto src_offset = [=]() -> PrimExpr {
            if (load->indices[0]->IsInstance<RampNode>()) {
              return load->indices[0].as<RampNode>()->base;
            }
            return PrimExpr();
          }();

          auto dst_offset = [=]() -> PrimExpr {
            if (store->indices[0].as<RampNode>()) {
              return store->indices[0].as<RampNode>()->base;
            } else if (store->indices[0].as<AddNode>()) {
              // The case where the dst buffer is a byte buffer generated by
              // merging dynamic shared memory. A_shared.dyn[(ramp(...), 1, 8) +
              // x8(17408))] = A_global[ramp(...),1, 8)]
              auto *add = store->indices[0].as<AddNode>();
              if (!add->a->IsInstance<RampNode>())
                return PrimExpr();
              if (!add->b->IsInstance<BroadcastNode>())
                return PrimExpr();
              return tir::Add(add->a.as<RampNode>()->base,
                              add->b.as<BroadcastNode>()->value);
            }
            return PrimExpr();
          }();

          if (src_offset.defined() && dst_offset.defined()) {
            // Calculate the number of elements based on bytes and dtype
            int dst_elem_count = bytes / dst_elem_type->bytes();
            int src_elem_count = bytes / src_elem_type->bytes();

            // Create access_ptr for destination (shared memory, write access)
            auto dst_access_ptr = store->buffer.access_ptr(
                2, DataType::Handle(), 1, dst_offset, PrimExpr(dst_elem_count));

            // Create access_ptr for source (global memory, read access)
            auto src_access_ptr = load->buffer.access_ptr(
                1, DataType::Handle(), 1, src_offset, PrimExpr(src_elem_count));

            ffi::Array<PrimExpr> cp_async_args{dst_access_ptr, src_access_ptr,
                                               PrimExpr(bytes)};
            bool lds_contiguous_vec = is_rocm_ && bytes == 16 &&
                                     IsLdsContiguous(dst_offset);
            const Op &op = lds_contiguous_vec
                               ? tl::ptx_cp_async_lds()
                               : tvm::tir::builtin::ptx_cp_async();
            return Evaluate(Call(store->buffer->dtype, op, cp_async_args));
          }
        } else {
          // Predicated vectorized cp.async - extract offsets from vectorized
          // indices
          auto src_offset = [=]() -> PrimExpr {
            if (load->indices[0]->IsInstance<RampNode>()) {
              return load->indices[0].as<RampNode>()->base;
            }
            return PrimExpr();
          }();

          auto dst_offset = [=]() -> PrimExpr {
            if (store->indices[0].as<RampNode>()) {
              return store->indices[0].as<RampNode>()->base;
            } else if (store->indices[0].as<AddNode>()) {
              // The case where the dst buffer is a byte buffer generated by
              // merging dynamic shared memory.
              auto *add = store->indices[0].as<AddNode>();
              if (!add->a->IsInstance<RampNode>())
                return PrimExpr();
              if (!add->b->IsInstance<BroadcastNode>())
                return PrimExpr();
              return tir::Add(add->a.as<RampNode>()->base,
                              add->b.as<BroadcastNode>()->value);
            }
            return PrimExpr();
          }();

          if (src_offset.defined() && dst_offset.defined()) {
            // Calculate the number of elements based on bytes and dtype
            int dst_elem_count = bytes / dst_elem_type->bytes();
            int src_elem_count = bytes / src_elem_type->bytes();

            // Create access_ptr for destination (shared memory, write access)
            auto dst_access_ptr = store->buffer.access_ptr(
                2, DataType::Handle(), 1, dst_offset, PrimExpr(dst_elem_count));

            // Create access_ptr for source (global memory, read access)
            auto src_access_ptr = load->buffer.access_ptr(
                1, DataType::Handle(), 1, src_offset, PrimExpr(src_elem_count));

            // Predicated vectorized cp.async with 4 arguments
            ffi::Array<PrimExpr> cp_async_args{dst_access_ptr, src_access_ptr,
                                               PrimExpr(bytes),
                                               predicate_value};
            return Evaluate(Call(store->buffer->dtype,
                                 tvm::tir::builtin::ptx_cp_async(),
                                 cp_async_args));
          } else {
            // If we can't extract offsets from vectorized indices, fall back
            LOG(WARNING)
                << "Cannot extract offsets from vectorized indices for "
                   "predicated cp.async, "
                << "falling back to regular buffer store/load";
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) {
    bool is_shared = (store->buffer.scope() == "shared" ||
                      store->buffer.scope() == "shared.dyn");
    if (in_async && is_shared) {
      if (auto *load = store->value.as<BufferLoadNode>()) {
        return InjectPTX(load, store);
      } else if (auto *call = store->value.as<CallNode>()) {
        // tir.if_then_else is a call to tir::builtin::if_then_else()
        if (call->op.same_as(builtin::if_then_else()) &&
            call->args.size() == 3) {
          if (auto *load = call->args[1].as<BufferLoadNode>()) {
            // Only default value of 0 is supported since 0 is the default value
            // used by cp.async ptx. @see section 9.7.8.22.3. of
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-memory-operations
            bool else_value_is_zero = false;
            if (auto *b = call->args[2].as<BroadcastNode>()) {
              if (auto *f = b->value.as<FloatImmNode>()) {
                else_value_is_zero = f->value == 0.0f;
              } else if (auto *i = b->value.as<IntImmNode>()) {
                else_value_is_zero = i->value == 0;
              }
            }
            if (auto *f = call->args[2].as<FloatImmNode>()) {
              else_value_is_zero = f->value == 0.0f;
            } else if (auto *i = call->args[2].as<IntImmNode>()) {
              else_value_is_zero = i->value == 0;
            }
            if (else_value_is_zero) {
              return InjectPTX(load, store, true, call->args[0]);
            }
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

private:
  bool in_async{false};
  bool is_rocm_{false};
};

using namespace tir::transform;

tvm::transform::Pass InjectPTXAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    bool is_rocm = false;
    if (target.defined()) {
      is_rocm = TargetIsRocm(target.value());
    } else {
      // Fallback: check Target::Current()
      auto current = Target::Current();
      if (current.defined()) {
        is_rocm = TargetIsRocm(current);
      }
    }
    auto *n = f.CopyOnWrite();
    n->body = PTXAsyncCopyInjector(is_rocm)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectPTXAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectPTXAsyncCopy", InjectPTXAsyncCopy);
}

} // namespace tl
} // namespace tvm
