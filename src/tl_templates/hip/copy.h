#pragma once

#include "common.h"

using f32 = float;
// using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

using ck_tile::int32x4_t;

// Address-space-3 (LDS) pointer type for buffer_load ... lds intrinsic
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

struct __attribute__((packed)) buffer_resource {
  const void *ptr;
  uint32_t range;
  uint32_t config;
};

CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void *ptr,
                                                   uint32_t size = 0xffffffff) {
  buffer_resource res{ptr, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
  int32x4_t r = __builtin_bit_cast(int32x4_t, res);
  r.x = __builtin_amdgcn_readfirstlane(r.x);
  r.y = __builtin_amdgcn_readfirstlane(r.y);
  r.z = __builtin_amdgcn_readfirstlane(r.z);
  r.w = __builtin_amdgcn_readfirstlane(r.w);
  return r;
}

__device__ void init_m0(uint32_t m0_value) {
  asm volatile("s_mov_b32 m0, %0" : : "s"(m0_value) : "memory");
}

__device__ void inc_m0(uint32_t m0_inc) {
  asm volatile("s_add_u32 m0, %0, m0" : : "n"(m0_inc) : "memory");
}

// LLVM intrinsic: async global-to-LDS transfer (gfx950+: supports 4/12/16 bytes)
// On gfx950, buffer_load_b128 ... lds: each lane loads N bytes from global,
// writes to LDS at m0 + lane_id * N.
__device__ void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

namespace tl {

// AMDGPU automatically commit memory fence
TL_DEVICE void cp_async_commit() {}

// Global Memory only fence
__device__ void async_gld_fence(index_t cnt) {
  asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// Global Memory and Shared Memory fence
__device__ void async_gld_sld_fence(index_t cnt) {
  asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
}

__device__ void wave_barrier() { asm volatile("s_barrier" : : : "memory"); }

template <int N = 0> TL_DEVICE void cp_async_wait() {
  async_gld_fence(N);
  // or
  // async_gld_sld_fence(N);
}

template <bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dword_v(void *smem, int32x4_t rsrc,
                                              index_t voffset) {
  auto const lds_ptr_sgpr =
      __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
  asm volatile("s_mov_b32 m0, %0; \n\t"
               "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
               "v"(voffset), "s"(rsrc)
               : "memory");
}

// ============================================================================
// Truly async global-to-LDS copy using buffer_load_b128 ... lds (gfx950+)
//
// Hardware behaviour per instruction:
//   - Each lane loads N bytes from global at: rsrc.base + soffset + voffset
//   - Each lane writes N bytes to LDS at:     m0 + lane_id * N
//   - One instruction moves 64 lanes * N bytes = 64*N bytes (1024B for N=16)
//   - Tracked by vmcnt (truly async, data bypasses VGPRs)
//
// cp_async_gs<N>(lds_ptr, global_ptr) keeps the original signature so that
// no IR/codegen changes are needed.  Internally it extracts the wave-uniform
// m0, buffer resource, and per-lane voffset from the two pointers.
//
// Prerequisites (guaranteed by the LowerTileOp swizzle-swap for ROCm):
//   - LDS addresses are contiguous per wavefront: lane k writes at
//     lds_base + k * N  (i.e. the swizzle has been moved to the load side).
// ============================================================================
template <int N>
TL_DEVICE void cp_async_gs(void *lds_base_ptr, void const *global_base_ptr) {
  if constexpr (N == 16) {
    // --- buffer_load_b128 ... lds  (truly async, bypasses VGPRs) ----------

    // m0 = wave-uniform LDS byte offset (lane 0's value).
    // Hardware writes to: m0 + lane_id * 16.
    uint32_t lds_m0 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds_base_ptr)));

    // Buffer resource descriptor from lane 0's global pointer.
    // make_wave_buffer_resource internally does readfirstlane on all
    // 128 bits, so rsrc.base = lane 0's global_base_ptr.
    auto rsrc = make_wave_buffer_resource(global_base_ptr);

    // Per-lane voffset (bytes) = this lane's global addr − lane 0's addr.
    // Only the low 32 bits are needed; within a wavefront the delta is
    // at most a few MB, so truncation is safe.
    uint32_t my_lo =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(global_base_ptr));
    uint32_t base_lo = __builtin_amdgcn_readfirstlane(my_lo);
    uint32_t voffset = my_lo - base_lo;

    // Issue the truly-async G2S transfer.
    asm volatile("s_mov_b32 m0, %0" : : "s"(lds_m0) : "memory");
    llvm_amdgcn_raw_buffer_load_lds(
        rsrc,
        (as3_uint32_ptr)0, // LDS base supplied via m0
        N,                 // bytes per lane (16)
        voffset,
        0,                 // soffset
        0,                 // immediate offset
        0                  // aux / cache policy
    );
  } else if constexpr (N == 8) {
    // No buffer_load ... lds variant for 8 bytes; use VGPR path.
    *(uint2 *)lds_base_ptr = *(const uint2 *)global_base_ptr;
  } else if constexpr (N == 4) {
    async_buffer_load_dword_v(
        lds_base_ptr,
        make_wave_buffer_resource(((const int32_t *)global_base_ptr) -
                                  threadIdx.x),
        threadIdx.x * N /*assume 4 bytes*/);
  }
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void *lds_base_ptr,
                                       void const *global_base_ptr, bool cond) {
  if constexpr (N == 16) {
    *(uint4 *)lds_base_ptr =
        cond ? *(const uint4 *)global_base_ptr : make_uint4(0, 0, 0, 0);
  } else if constexpr (N == 8) {
    *(uint2 *)lds_base_ptr =
        cond ? *(const uint2 *)global_base_ptr : make_uint2(0, 0);
  } else {
    if (cond) {
      async_buffer_load_dword_v(
          lds_base_ptr,
          make_wave_buffer_resource(((const int32_t *)global_base_ptr) -
                                    threadIdx.x),
          threadIdx.x * N /*assume 4 bytes*/);
    } else {
      *(uint4 *)lds_base_ptr = make_uint4(0, 0, 0, 0);
    }
  }
}

} // namespace tl
