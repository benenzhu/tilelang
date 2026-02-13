/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_HIP_H_
#define TVM_TL_TARGET_CODEGEN_HIP_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "target/source/codegen_c.h"

namespace tvm{
namespace codegen {

class CodeGenTileLangHIP final : public CodeGenC {
public:
  CodeGenTileLangHIP();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const PrimFunc &f);

  // Extract the global buffer data Var from a tvm_access_ptr expression.
  // Public because G2SBufferScanner (in codegen_hip.cc) needs access.
  static const tir::VarNode *
  ExtractGlobalBufVar(const PrimExpr &access_ptr_expr);

protected:
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangHIP *p);

  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mfma.h
  bool need_wmma_h_{false};
  // whether need fp8.h
  bool enable_fp8_{false};
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  // --- Async G2S pipeline tracking for AMD vmcnt ---
  // On AMD, vmcnt counts individual instructions (not groups like NVIDIA).
  // We track how many ptx_cp_async ops are issued per commit group so that
  // ptx_wait_group(N) can be emitted as vmcnt(N * ops_per_group).
  // Loop trip counts are tracked to account for unrolled loops containing
  // async ops (the IR visits the op once, but it executes trip_count times).
  int async_ops_since_commit_{0};
  int ops_per_commit_group_{0};
  std::vector<int> loop_trip_counts_;

  // --- Optimized G2S copy with precomputed SRD (AMD) ---
  // Maps global buffer VarNode* to the name of the hoisted SRD variable.
  // Populated during AddFunction's pre-scan, used when emitting ptx_cp_async.
  std::unordered_map<const tir::VarNode *, std::string> g2s_srd_map_;

  // Pre-scan TIR body to collect global buffer Vars used in ptx_cp_async calls.
  void ScanForG2SBuffers(const tir::Stmt &body);

  // Emit a decomposed G2S copy using a precomputed SRD.
  // When hoisted G2S groups are available (precomputed before the pipeline
  // loop), uses the precomputed voffset/soffset arrays.  Otherwise falls
  // back to inline computation.
  void EmitDecomposedG2S(const std::string &dst, const std::string &src,
                         const std::string &size,
                         const tir::VarNode *buf_var,
                         const PrimExpr &src_access_ptr,
                         const PrimExpr &dst_access_ptr);

  // --- Pipeline loop tracking for G2S voffset hoisting ---
  // The outermost Serial for-loop variable (the k-loop in pipelined GEMM).
  tir::Var pipeline_loop_var_;
  bool in_pipeline_loop_{false};

  // Precomputed G2S voffset/soffset groups, emitted before the pipeline loop.
  // Each entry corresponds to one ptx_cp_async call site (in DFS visit order).
  struct G2SHoistGroup {
    std::string voff_name;      // precomputed voffset array/scalar name
    std::string soff_name;      // precomputed soff_base array/scalar name
    std::string k_stride_str;   // k coefficient in bytes, as C expression
    tir::Var unroll_var;        // enclosing unrolled loop var (undef if none)
    int64_t unroll_extent;      // 0 if not in unrolled loop
  };
  std::vector<G2SHoistGroup> g2s_hoist_groups_;
  int g2s_hoist_emit_idx_{0};

  // Scan pipeline loop body and emit precomputed voffset/soffset arrays.
  void EmitG2SHoistPrecomputation(const tir::ForNode *pipeline_loop);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_HIP_H_
