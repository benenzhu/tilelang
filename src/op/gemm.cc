// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/gemm.cc
 *
 * Define gemm operator.
 */

#include "gemm.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

static std::vector<int> toPrimeFactors(int x) {
  int i = 2;
  std::vector<int> result;
  while (x > 1) {
    if (x % i == 0) {
      x /= i;
      result.push_back(i);
    } else {
      i++;
    }
  }
  return result;
}

Gemm::Gemm(Array<PrimExpr> args, BufferMap vmap) {
  A = vmap[GetVarFromAccessPtr(args[0])];
  B = vmap[GetVarFromAccessPtr(args[1])];
  C = vmap[GetVarFromAccessPtr(args[2])];
  trans_A = args[3].as<Bool>().value();
  trans_B = args[4].as<Bool>().value();
  M = args[5].as<IntImm>().value()->value;
  N = args[6].as<IntImm>().value()->value;
  K = args[7].as<IntImm>().value()->value;
  policy = static_cast<GemmWarpPolicy>(args[8].as<IntImm>().value()->value);
  clear_accum = args[9].as<Bool>().value();
  if (args.size() > 10) {
    kPack = args[10].as<IntImm>().value()->value;
    if (kPack != 1 && kPack != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 11) {
    wg_wait = args[11].as<IntImm>().value()->value;
  }
}

std::pair<int, int> Gemm::ComputeWarpPartition(int num_warps, Target target,
                                               bool maybe_hopper_wgmma) const {
  int m_warp = 1, n_warp = 1;
  bool allow_wgmma = TargetIsHopper(target) && maybe_hopper_wgmma &&
                     (this->M >= 64) && (num_warps % 4 == 0);
  if (allow_wgmma) {
    ICHECK(num_warps % 4 == 0) << "Use Warp Group MMA requires 128*N threads.";
    if (this->policy == GemmWarpPolicy::kFullRow ||
        this->policy == GemmWarpPolicy::kSquare) {
      m_warp = num_warps;
      ICHECK(this->M % num_warps == 0);
    } else if (this->policy == GemmWarpPolicy::kFullCol) {
      m_warp = 4;
      n_warp = num_warps / 4;
      ICHECK(this->N % n_warp == 0);
    } else {
      ICHECK(0) << "Unknown GemmWarpPolicy";
    }
    return {m_warp, n_warp};
  }
  if (this->policy == GemmWarpPolicy::kFullRow) {
    m_warp = num_warps;
    ICHECK(this->M % num_warps == 0);
  } else if (this->policy == GemmWarpPolicy::kFullCol) {
    n_warp = num_warps;
    ICHECK(this->N % num_warps == 0);
  } else if (this->policy == GemmWarpPolicy::kSquare) {
    auto factors = toPrimeFactors(num_warps);
    for (int factor : factors) {
      bool M_divisible = (this->M % (factor * m_warp)) == 0;
      bool N_divisible = (this->N % (factor * n_warp)) == 0;
      if (M_divisible && N_divisible) {
        // put N dimension first
        // because usually n in mma
        // is more smaller than m
        if (this->N / n_warp >= this->M / m_warp)
          n_warp *= factor;
        else
          m_warp *= factor;
      } else if (N_divisible) {
        n_warp *= factor;
      } else if (M_divisible) {
        m_warp *= factor;
      } else {
        ICHECK(0) << "Cannot compute warp partition for shape" << M << " " << N
                  << " with num_warps " << num_warps;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }
  return {m_warp, n_warp};
}

Stmt Gemm::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  int warp_size = 32;
  if (TargetIsCDNA(T.target)) {
    warp_size = 64;
  }
  auto block_size = *as_const_int(T.thread_bounds->extent);
  bool maybe_wgmma = TargetIsHopper(T.target) && (this->M >= 64) &&
                     (block_size / warp_size % 4 == 0);

  auto [warp_m, warp_n] =
      ComputeWarpPartition(block_size / warp_size, T.target, maybe_wgmma);

  std::stringstream ss;
  std::string op_name = "tl::gemm_ss";
  if (A.scope() == "local.fragment") {
    ICHECK(B.scope() != "local.fragment");
    op_name = "tl::gemm_rs";
  } else if (B.scope() == "local.fragment") {
    op_name = "tl::gemm_sr";
  }
  ss << op_name << "<" << M << ", " << N << ", " << K << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << trans_A << ", " << trans_B;
  ss << ", " << clear_accum;
  if (TargetIsCDNA(T.target)) {
    // for cdna gemm, we need to specify kPack
    ss << ", " << kPack;
  } else if (TargetIsHopper(T.target)) {
    ss << ", " << (maybe_wgmma ? "true" : "false");
  }
  if (wg_wait != 0) {
    ss << ", " << wg_wait;
  }
  ss << ">";
  auto A_buffer = T.buffer_remap.count(A) ? T.buffer_remap[A] : A;
  auto B_buffer = T.buffer_remap.count(B) ? T.buffer_remap[B] : B;
  auto C_buffer = T.buffer_remap[C];

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(A_buffer.access_ptr(1));
  new_args.push_back(B_buffer.access_ptr(1));
  new_args.push_back(C_buffer.access_ptr(3));
  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap Gemm::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  if (completed_)
    return {};
  LayoutMap results;
  ICHECK(C.scope() == "local.fragment");
  auto block_size = *as_const_int(T.thread_bounds->extent) -
                    *as_const_int(T.thread_bounds->min);
  if (TargetIsVolta(T.target)) {
    const int warp_size = 32;
    auto [warp_m, warp_n] =
        ComputeWarpPartition(block_size / warp_size, T.target);
    auto fragment =
        makeGemmVoltaFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment);
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      results.Set(A, makeGemmVoltaABLayout(*as_const_int(A->shape[0]),
                                           *as_const_int(A->shape[1]), true,
                                           trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      ICHECK(trans_A == false);
      results.Set(A, makeGemmVoltaFragmentA(M, N, K, M / warp_m, N / warp_n));
    } else {
      ICHECK(0);
    }

    ICHECK(B.scope() == "shared" || B.scope() == "shared.dyn");
    results.Set(B, makeGemmVoltaABLayout(*as_const_int(B->shape[0]),
                                         *as_const_int(B->shape[1]), false,
                                         trans_B ? 2 : 1));
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target)) {
    const int warp_size = 32;
    auto [warp_m, warp_n] =
        ComputeWarpPartition(block_size / warp_size, T.target);
    auto fragment =
        makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment);

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      const int64_t mat_stride = *as_const_int(A->shape[0]);
      const int64_t mat_continuous = *as_const_int(A->shape[1]);
      results.Set(A,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   A->dtype.bits(), trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      ICHECK(trans_A == false);
      results.Set(A, makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
                                       A->dtype.bits()));
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      const int64_t mat_stride = *as_const_int(B->shape[0]);
      const int64_t mat_continuous = *as_const_int(B->shape[1]);
      results.Set(B,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   B->dtype.bits(), trans_B ? 2 : 1));
    } else if (B.scope() == "local.fragment") {
      ICHECK(trans_B == false) << "B is local.fragment, trans_B must be false, "
                                  "please raise an issue if you see this";
      results.Set(B, makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsHopper(T.target)) {
    const int warp_size = 32;
    bool maybe_wgmma = (this->M >= 64) && (block_size / warp_size % 4 == 0);
    auto [warp_m, warp_n] =
        ComputeWarpPartition(block_size / warp_size, T.target, maybe_wgmma);
    auto fragment =
        maybe_wgmma
            ? makeGemmFragmentCHopper(M, N, M / warp_m, N / warp_n,
                                      C->dtype.bits())
            : makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment);
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      const int64_t mat_stride = *as_const_int(A->shape[0]);
      const int64_t mat_continuous = *as_const_int(A->shape[1]);
      const int64_t continuity =
          trans_A ? mat_continuous / (warp_m / 4) : mat_continuous;
      results.Set(A, makeGemmABLayout(mat_stride, mat_continuous, continuity,
                                      A->dtype.bits(), trans_A ? 1 : 2));
    } else {
      ICHECK(trans_A == false);
      results.Set(A, makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
                                       A->dtype.bits()));
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      const int64_t mat_stride = *as_const_int(B->shape[0]);
      const int64_t mat_continuous = *as_const_int(B->shape[1]);
      const int64_t continuity =
          trans_B ? mat_continuous : mat_continuous / warp_n;
      results.Set(B, makeGemmABLayout(mat_stride, mat_continuous, continuity,
                                      B->dtype.bits(), trans_B ? 2 : 1));
    } else {
      ICHECK(0) << "WGMMA only support B in shared.";
    }
  } else if (TargetIsCDNA(T.target)) {
    const int warp_size = 64;
    auto [warp_m, warp_n] =
        ComputeWarpPartition(block_size / warp_size, T.target);

    auto fragment =
        makeGemmFragmentCCDNA(M, N, M / warp_m, N / warp_n, C->dtype.bits());

    results.Set(C, fragment);

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {

      // Make Linear Memory Access Layout
      // auto shared_layout =
      //     makeGemmLayoutLinear(*as_const_int(A->shape[0]),
      //     *as_const_int(A->shape[1]));

      // Make Swizzle or Pad Layout
      auto shared_layout = makeGemmABLayoutCDNA(*as_const_int(A->shape[0]),
                                                *as_const_int(A->shape[1]),
                                                A->dtype.bits(), kPack);
      results.Set(A, shared_layout);
    } else if (A.scope() == "local.fragment") {
      results.Set(A, makeGemmFragmentACDNA(M, N, K, M / warp_m, N / warp_n,
                                           A->dtype.bits(), trans_A));
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      // Make Linear Memory Access Layout
      // auto shared_layout =
      //     makeGemmLayoutLinear(*as_const_int(B->shape[0]),
      //     *as_const_int(B->shape[1]));

      // Make Swizzle or Pad Layout
      auto shared_layout = makeGemmABLayoutCDNA(*as_const_int(B->shape[0]),
                                                *as_const_int(B->shape[1]),
                                                B->dtype.bits(), kPack);

      results.Set(B, shared_layout);
    } else if (B.scope() == "local.fragment") {
      results.Set(B, makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n));
    } else {
      ICHECK(0);
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str();
  }
  completed_ = true;
  return results;
}

TIR_REGISTER_TL_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm