/*!
 * \file tl/backend/rocm/op/copy.cc
 * \brief ROCm implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/ptx_async_copy_injector.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

namespace {

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

} // namespace

// ---------------------------------------------------------------------------
// Swizzle-swap mutator: rewrite g2s BufferStore to land at lane-contiguous
// LDS addresses while reflecting the XOR delta to the global side. This is
// what makes the gfx950 `buffer_load_dwordx4 ... lds` path safe to use.
//
// Input (post-LowerParallelLoop, pre-InjectPTXAsyncCopy):
//
//     shared_orig[s, m, k] = global_orig[g_row(s,m,k), g_col(s,m,k)]
//
// Output (still pre-Inject):
//
//     shared_remapped[Forward(s,m,k) with last dim -= delta(s,m,k)]
//         = global_orig[g_row(s,m,k), g_col(s,m,k) + delta(s,m,k)]
//
// Using the remapped buffer directly defeats the later
// RemapBufferRewriter Forward-apply (it skips already-remapped buffers).
// ---------------------------------------------------------------------------
class SwizzleSwapMutator : public StmtExprMutator {
public:
  SwizzleSwapMutator(const LayoutMap &layout_map,
                     const Map<Buffer, Buffer> &buffer_remap,
                     arith::Analyzer *analyzer)
      : layout_map_(layout_map),
        buffer_remap_(buffer_remap),
        analyzer_(analyzer) {}

  bool engaged() const { return engaged_; }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    // Bind the loop var range so analyzer_->Simplify can fold things like
    // (vec % 8) -> vec when vec is bound to [0, 8).
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    Buffer buffer = store->buffer;

    if (!IsSharedBuffer(buffer))
      return store;
    auto layout_it = layout_map_.find(buffer);
    if (layout_it == layout_map_.end())
      return store;
    Layout layout = (*layout_it).second;
    if (!layout->HasSwizzle())
      return store;
    auto remap_it = buffer_remap_.find(buffer);
    if (remap_it == buffer_remap_.end())
      return store;
    Buffer new_buffer = (*remap_it).second;

    const BufferLoadNode *load = store->value.as<BufferLoadNode>();
    if (load == nullptr)
      return store;
    if (!IsGlobalBuffer(load->buffer))
      return store;
    if (load->indices.empty())
      return store;

    // Compute the swizzled (Forward'd) store indices and the per-element
    // XOR delta. Layout->Forward takes the full store->indices (the
    // leading dim is just the pipeline stage and passes through unchanged).
    Array<PrimExpr> forwarded = layout->Forward(store->indices);
    PrimExpr delta = analyzer_->Simplify(layout->SwizzleDelta(store->indices));
    if (forwarded.empty())
      return store;

    // LDS write index: drop the XOR from the last forwarded dim. The result
    // is the lane-contiguous physical position (base + lane * vec_size).
    Array<PrimExpr> sequential_store(forwarded.begin(), forwarded.end());
    int last_out = static_cast<int>(forwarded.size()) - 1;
    sequential_store.Set(
        last_out, analyzer_->Simplify(forwarded[last_out] - delta));

    // Global load index: add the XOR delta to the last dim of the global
    // index (the column dim, where the swizzle lives in the shared layout).
    Array<PrimExpr> new_load_indices(load->indices.begin(),
                                     load->indices.end());
    int last_load = static_cast<int>(load->indices.size()) - 1;
    new_load_indices.Set(
        last_load, analyzer_->Simplify(load->indices[last_load] + delta));

    engaged_ = true;
    // Use the remapped (swizzle-shaped) buffer with our linear indices --
    // this prevents the later RemapBufferRewriter from re-applying Forward
    // (which would re-introduce the XOR we just removed).
    return BufferStore(new_buffer,
                       BufferLoad(load->buffer, new_load_indices),
                       sequential_store);
  }

  const LayoutMap &layout_map_;
  const Map<Buffer, Buffer> &buffer_remap_;
  arith::Analyzer *analyzer_;
  bool engaged_{false};
};

enum class CopyInst : uint8_t {
  kNormal = 0,
  kCPAsync = 1,
};

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    SelectInst(op, T.target, T.layout_map, T.analyzer);
    return op.InferSIMTLayout(T, level);
  }

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer) {
    if (GetIsAsyncCopy(op) || GetNoImplicitAsyncCommitWait(op)) {
      bool cp_async_supported =
          CheckCPAsyncCopy(op, target, layout_map, analyzer);
      ICHECK(cp_async_supported)
          << "Explicit async copy semantics require ROCm async copy lowering, "
             "but constraints were not satisfied. Got src="
          << op.src->name << " (scope=" << op.src.scope()
          << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
          << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype
          << ").";
      return CopyInst::kCPAsync;
    }
    return CopyInst::kNormal;
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    auto copy_inst = SelectInst(op, T.target, T.layout_map, analyzer);
    if (copy_inst == CopyInst::kCPAsync) {
      return LowerCPAsync(op, T, analyzer);
    }
    if (copy_inst == CopyInst::kNormal) {
      return LowerNormalCopy(op, T, analyzer);
    }
    LOG(FATAL) << "Unsupported ROCm copy inst " << static_cast<int>(copy_inst);
  }

private:
  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &T,
                           arith::Analyzer *analyzer) {
    using namespace tvm::transform;

    PassContext pass_ctx = PassContext::Current();
    bool enable_async_copy =
        pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
    bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait(op);
    bool explicit_async_semantics =
        no_implicit_commit_wait || GetIsAsyncCopy(op);
    if (!enable_async_copy && !explicit_async_semantics) {
      return LowerNormalCopy(op, T, analyzer);
    }

    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);

    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    Stmt lowered_loop = LowerParallelLoop(par_op->GetRoot(), loop_layout,
                                          T.thread_var, analyzer, T.layout_map,
                                          par_op->GetPredicate(T.thread_var));

    // Pre-bind the thread var range so the swap's analyzer can simplify
    // expressions like (tx % 32 // 16 + ...) cleanly.
    analyzer->Bind(T.thread_var, T.thread_bounds);

    // ROCm-only: rewrite the g2s BufferStore so LDS writes are
    // lane-contiguous (precondition for buffer_load_dwordx4 ... lds).
    //
    // OFF BY DEFAULT for now: the swap is semantically correct (verified
    // on 1024^3 NT) but the indices it produces for some layouts (e.g.,
    // the NN B-matrix tile where the inner row is the wide N dim) defeat
    // the InjectPTXAsyncCopy vectorization detector and trigger a
    // "ptx_cp_async requires byte width in {4,8,16}, but got 2" abort.
    // Enable explicitly with TL_ENABLE_ROCM_SWIZZLE_SWAP=1.
    const char *swap_env = std::getenv("TL_ENABLE_ROCM_SWIZZLE_SWAP");
    bool enable_swap = swap_env && std::string(swap_env) != "0";
    Stmt swapped_loop = lowered_loop;
    if (enable_swap) {
      SwizzleSwapMutator swap_mutator(T.layout_map, T.buffer_remap, analyzer);
      swapped_loop = swap_mutator(lowered_loop);
    }

    auto inject_result =
        InjectPTXAsyncCopy(swapped_loop, /*enable_auto_async_copy=*/true,
                           /*async_without_async_commit_wait=*/
                           no_implicit_commit_wait || GetIsAsyncCopy(op));
    Stmt cp_async_loop = inject_result.stmt;
    if (!inject_result.injected_ptx_async_copy) {
      DLOG(WARNING) << "cp.async rewrite miss for copy src=" << op.src->name
                    << " (scope=" << op.src.scope()
                    << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
                    << " (scope=" << op.dst.scope()
                    << ", dtype=" << op.dst->dtype
                    << "), no_implicit_async_commit_wait="
                    << no_implicit_commit_wait
                    << ", is_async_copy=" << GetIsAsyncCopy(op);
      if (no_implicit_commit_wait) {
        DLOG(WARNING)
            << "Pipeline-managed async copy fallback to normal copy because "
               "cp.async rewrite found no eligible global->shared store.";
        return lowered_loop;
      }
      if (explicit_async_semantics) {
        LOG(FATAL)
            << "Explicit async copy semantics require cp.async lowering, "
               "but no eligible global->shared store was rewritten.";
      }
      DLOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                       "no eligible global->shared store.";
      return LowerNormalCopy(op, T, analyzer);
    }
    if (no_implicit_commit_wait) {
      return cp_async_loop;
    }
    if (GetIsAsyncCopy(op)) {
      Stmt commit_group =
          Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
      return SeqStmt({cp_async_loop, commit_group});
    }
    return cp_async_loop;
  }

  static bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
    if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
      return false;
    }
    if (op.src->dtype != op.dst->dtype) {
      return false;
    }
    return true;
  }

  static bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) {
    if (!TargetHasAsyncCopy(target)) {
      return false;
    }
    return CheckCPAsyncCopyPreconditions(op);
  }
};

} // namespace rocm

namespace {

bool MatchROCmCopyTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmCopy() {
  RegisterCopyImpl(CopyImpl{
      "rocm.Copy",
      MatchROCmCopyTarget,
      100,
      rocm::Copy::InferLayout,
      rocm::Copy::Lower,
  });
  return true;
}

const bool rocm_copy_registered = RegisterROCmCopy();

} // namespace

} // namespace tl
} // namespace tvm
