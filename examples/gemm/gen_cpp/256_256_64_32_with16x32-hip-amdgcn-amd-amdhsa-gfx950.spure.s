	.file	0 "/root/learn-hip/HipKittens/kernels/gemm/bf16fp32" "256_256_64_32_with16x32.cpp" md5 0xfd55c1f59cd8cc8ead3fac5f2efa1246
	.file	1 "../../..//include/common" "util.cuh"
	.file	2 "256_256_64_32_with16x32.cpp"
	.file	3 "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/ext" "concurrence.h"
	.file	4 "../../..//include/ops/warp/memory/util" "util.cuh"
	.file	5 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_bf16.h"
	.file	6 "../../..//include/common" "base_types.cuh"
	.file	7 "/usr/include/x86_64-linux-gnu/bits" "types.h"
	.file	8 "/usr/include/x86_64-linux-gnu/bits" "stdint-uintn.h"
	.file	9 "/usr/include" "stdint.h"
	.file	10 "/usr/include/x86_64-linux-gnu/bits" "stdint-intn.h"
	.file	11 "../../..//include/types/shared" "st_shape.cuh"
	.file	12 "../../..//include/types/shared" "st.cuh"
	.file	13 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_vector_types.h"
	.file	14 "/opt/rocm-7.2.0/lib/llvm/lib/clang/20/include" "__stddef_size_t.h"
	.file	15 "../../..//include/ops/warp/memory/tile" "global_to_shared.cuh"
	.file	16 "../../..//include/ops/warp/register/tile" "mma.cuh"
	.file	17 "../../..//include/types/global" "gl.cuh"
	.file	18 "../../..//include/ops/warp/memory/tile" "global_to_register.cuh"
	.file	19 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip" "hip_runtime_api.h"
	.file	20 "../../..//include/types/register" "rt_shape.cuh"
	.file	21 "../../..//include/types/register" "rt.cuh"
	.file	22 "../../..//include/types/register" "rt_base.cuh"
	.file	23 "../../..//include/types/global" "util.cuh"
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z8micro_tk13micro_globalsiii ; -- Begin function _Z8micro_tk13micro_globalsiii
	.globl	_Z8micro_tk13micro_globalsiii
	.p2align	8
	.type	_Z8micro_tk13micro_globalsiii,@function
_Z8micro_tk13micro_globalsiii:          ; @_Z8micro_tk13micro_globalsiii
	s_load_dwordx2 s[16:17], s[0:1], 0x20	;.loc	24 0 115 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:0:115
	s_load_dwordx2 s[8:9], s[0:1], 0x30
	s_load_dwordx2 s[18:19], s[0:1], 0x50
	s_waitcnt lgkmcnt(0)	;.loc	2 57 28 is_stmt 1               ; 256_256_64_32_with16x32.cpp:57:28
	s_mul_i32 s3, s10, s3
	s_add_i32 s6, s3, s2	;.loc	2 57 41 is_stmt 0               ; 256_256_64_32_with16x32.cpp:57:41
	s_mul_i32 s2, s11, s10	;.loc	2 58 35 is_stmt 1               ; 256_256_64_32_with16x32.cpp:58:35
	s_ashr_i32 s3, s2, 31	;.loc	1 89 33                         ; ../../..//include/common/util.cuh:89:33
	s_lshr_b32 s3, s3, 23
	s_add_i32 s2, s2, s3
	s_and_b32 s2, s2, 0xfffffe00	;.loc	1 89 42 is_stmt 0               ; ../../..//include/common/util.cuh:89:42
	s_cmp_gt_i32 s6, s2	;.loc	1 92 22 is_stmt 1               ; ../../..//include/common/util.cuh:92:22
.JUMP.LBB0_2:
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s2, s6, 31	;.loc	1 95 34                         ; ../../..//include/common/util.cuh:95:34
	s_lshr_b32 s3, s2, 29
	s_add_i32 s3, s6, s3
	s_ashr_i32 s7, s3, 3
	s_and_b32 s10, s3, 0x3fffff8	;.loc	1 85 28                         ; ../../..//include/common/util.cuh:85:28
	s_ashr_i32 s3, s3, 31	;.loc	1 97 34                         ; ../../..//include/common/util.cuh:97:34
	s_lshr_b32 s2, s2, 23	;.loc	1 96 31                         ; ../../..//include/common/util.cuh:96:31
	s_lshr_b32 s3, s3, 26	;.loc	1 97 34                         ; ../../..//include/common/util.cuh:97:34
	s_sub_i32 s10, s6, s10	;.loc	1 85 28                         ; ../../..//include/common/util.cuh:85:28
	s_add_i32 s2, s6, s2	;.loc	1 96 31                         ; ../../..//include/common/util.cuh:96:31
	s_add_i32 s3, s7, s3	;.loc	1 97 34                         ; ../../..//include/common/util.cuh:97:34
	s_andn2_b32 s3, s3, 63
	s_and_b32 s2, s2, 0xfffffe00	;.loc	1 100 22                        ; ../../..//include/common/util.cuh:100:22
	s_lshl_b32 s6, s10, 6	;.loc	1 100 36 is_stmt 0              ; ../../..//include/common/util.cuh:100:36
	s_sub_i32 s3, s7, s3	;.loc	1 97 34 is_stmt 1               ; ../../..//include/common/util.cuh:97:34
	s_add_i32 s2, s2, s6	;.loc	1 100 30                        ; ../../..//include/common/util.cuh:100:30
	s_add_i32 s6, s2, s3	;.loc	1 100 49 is_stmt 0              ; ../../..//include/common/util.cuh:100:49
.LBB0_2:                                ; %_ZN7kittens25chiplet_transform_chunkedEiiii.exit
	s_mov_b64 s[10:11], src_shared_base	;.loc	1 287 22 is_stmt 1              ; ../../..//include/common/util.cuh:287:22
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s10, 0, 0
	s_cselect_b32 s7, s11, 0
	s_and_b32 s2, s10, 15
	s_and_b32 s11, s10, -16	;.loc	1 287 34 is_stmt 0              ; ../../..//include/common/util.cuh:287:34
	s_load_dwordx4 s[12:15], s[0:1], 0xa8
	s_add_u32 s11, s11, 16
	s_mov_b32 s3, 0	;.loc	1 287 22                        ; ../../..//include/common/util.cuh:287:22
	s_waitcnt lgkmcnt(0)	;.loc	1 287 34                        ; ../../..//include/common/util.cuh:287:34
	s_addc_u32 s15, s7, 0
	s_cmp_eq_u64 s[2:3], 0
	s_cselect_b32 s20, s10, s11
	s_cselect_b32 s2, s7, s15
	s_add_u32 s7, s20, 0x10000	;.loc	1 311 17 is_stmt 1              ; ../../..//include/common/util.cuh:311:17
	s_and_b32 s10, s7, -16	;.loc	1 287 34                        ; ../../..//include/common/util.cuh:287:34
	s_and_b32 s2, s7, 15	;.loc	1 287 22 is_stmt 0              ; ../../..//include/common/util.cuh:287:22
	s_add_u32 s10, s10, 16	;.loc	1 287 34                        ; ../../..//include/common/util.cuh:287:34
	s_cmp_eq_u64 s[2:3], 0
	s_cselect_b32 s21, s7, s10
	s_add_i32 s2, s13, 0xff	;.loc	1 67 19 is_stmt 1               ; ../../..//include/common/util.cuh:67:19
	s_ashr_i32 s7, s2, 31	;.loc	1 67 24 is_stmt 0               ; ../../..//include/common/util.cuh:67:24
	s_lshr_b32 s7, s7, 24
	s_add_i32 s2, s2, s7
	s_ashr_i32 s2, s2, 8
	s_lshl_b32 s7, s2, 3	;.loc	2 65 39 is_stmt 1               ; 256_256_64_32_with16x32.cpp:65:39
	s_abs_i32 s10, s7	;.loc	2 66 25                         ; 256_256_64_32_with16x32.cpp:66:25
	v_cvt_f32_u32_e32 v001, s10
	s_add_i32 s11, s12, 0xff	;.loc	1 67 19                         ; ../../..//include/common/util.cuh:67:19
	s_sub_i32 s17, 0, s10	;.loc	2 66 25                         ; 256_256_64_32_with16x32.cpp:66:25
	s_ashr_i32 s15, s11, 31	;.loc	1 67 24                         ; ../../..//include/common/util.cuh:67:24
	v_rcp_iflag_f32_e32 v001, v001	;.loc	2 66 25                         ; 256_256_64_32_with16x32.cpp:66:25
	s_lshr_b32 s15, s15, 24	;.loc	1 67 24                         ; ../../..//include/common/util.cuh:67:24
	s_add_i32 s11, s11, s15
	s_abs_i32 s15, s6	;.loc	2 66 25                         ; 256_256_64_32_with16x32.cpp:66:25
	v_mul_f32_e32 v001, 0x4f7ffffe, v001
	v_cvt_u32_f32_e32 v001, v001
	s_xor_b32 s2, s6, s2
	s_ashr_i32 s11, s11, 8	;.loc	1 67 24                         ; ../../..//include/common/util.cuh:67:24
	s_ashr_i32 s2, s2, 31	;.loc	2 66 25                         ; 256_256_64_32_with16x32.cpp:66:25
	v_readfirstlane_b32 s19, v001
	s_mul_i32 s17, s17, s19
	s_mul_hi_u32 s17, s19, s17
	s_add_i32 s19, s19, s17
	s_mul_hi_u32 s17, s15, s19
	s_mul_i32 s19, s17, s10
	s_sub_i32 s15, s15, s19
	s_add_i32 s19, s17, 1
	s_sub_i32 s22, s15, s10
	s_cmp_ge_u32 s15, s10
	s_cselect_b32 s17, s19, s17
	s_cselect_b32 s15, s22, s15
	s_add_i32 s19, s17, 1
	s_cmp_ge_u32 s15, s10
	s_cselect_b32 s10, s19, s17
	s_xor_b32 s10, s10, s2
	s_sub_i32 s2, s10, s2
	s_lshl_b32 s10, s2, 3	;.loc	2 67 32                         ; 256_256_64_32_with16x32.cpp:67:32
	s_sub_i32 s11, s11, s10	;.loc	2 68 38                         ; 256_256_64_32_with16x32.cpp:68:38
	.file	25 "/opt/rocm-7.2.0/lib/llvm/lib/clang/20/include" "__clang_hip_math.h"
	s_min_i32 s11, s11, 8	;.loc	25 1325 10                      ; /opt/rocm-7.2.0/lib/llvm/lib/clang/20/include/__clang_hip_math.h:1325:10
	s_abs_i32 s15, s11	;.loc	2 70 44                         ; 256_256_64_32_with16x32.cpp:70:44
	v_cvt_f32_u32_e32 v001, s15
	s_sub_i32 s17, 0, s15
	s_mul_i32 s2, s2, s7	;.loc	2 69 38                         ; 256_256_64_32_with16x32.cpp:69:38
	s_sub_i32 s2, s6, s2
	v_rcp_iflag_f32_e32 v001, v001	;.loc	2 70 44                         ; 256_256_64_32_with16x32.cpp:70:44
	s_abs_i32 s7, s2
	s_xor_b32 s6, s2, s11
	s_ashr_i32 s6, s6, 31
	v_mul_f32_e32 v001, 0x4f7ffffe, v001
	v_cvt_u32_f32_e32 v001, v001
	v_lshrrev_b32_e32 v002, 6, v000	;.loc	1 39 97                         ; ../../..//include/common/util.cuh:39:97
	v_lshlrev_b32_e32 v003, 10, v002	;.loc	2 93 131                        ; 256_256_64_32_with16x32.cpp:93:131
	v_lshlrev_b32_e32 v004, 4, v000
	v_readfirstlane_b32 s19, v001	;.loc	2 70 44                         ; 256_256_64_32_with16x32.cpp:70:44
	s_mul_i32 s17, s17, s19
	s_mul_hi_u32 s17, s19, s17
	s_add_i32 s19, s19, s17
	s_mul_hi_u32 s17, s7, s19
	s_mul_i32 s19, s17, s15
	s_sub_i32 s7, s7, s19
	s_add_i32 s19, s17, 1
	s_sub_i32 s22, s7, s15
	s_cmp_ge_u32 s7, s15
	s_cselect_b32 s17, s19, s17
	s_cselect_b32 s7, s22, s7
	s_add_i32 s19, s17, 1
	s_cmp_ge_u32 s7, s15
	s_cselect_b32 s7, s19, s17
	s_xor_b32 s7, s7, s6
	s_sub_i32 s15, s7, s6
	s_mul_i32 s6, s15, s11	;.loc	2 69 59                         ; 256_256_64_32_with16x32.cpp:69:59
	s_sub_i32 s17, s2, s6
	s_add_i32 s17, s17, s10	;.loc	2 69 29 is_stmt 0               ; 256_256_64_32_with16x32.cpp:69:29
	s_lshl_b32 s10, s16, 17	;.loc	2 0 29                          ; 256_256_64_32_with16x32.cpp:0:29
	s_mul_i32 s2, s16, s12	;.loc	2 88 47 is_stmt 1               ; 256_256_64_32_with16x32.cpp:88:47
	s_or_b32 s11, s10, -2.0
	s_mov_b32 s10, s3	;.loc	2 0 47 is_stmt 0                ; 256_256_64_32_with16x32.cpp:0:47
	s_lshl_b32 s6, s2, 1	;.loc	2 88 47                         ; 256_256_64_32_with16x32.cpp:88:47
	s_and_b32 s2, s16, 0x1fff	;.loc	4 78 9 is_stmt 1                ; ../../..//include/ops/warp/memory/util/util.cuh:78:9
	s_or_b64 s[10:11], s[10:11], s[4:5]
	s_cmp_eq_u32 s2, 0
	s_mul_i32 s2, s18, s13	;.loc	2 89 47                         ; 256_256_64_32_with16x32.cpp:89:47
	s_cselect_b32 s5, s5, s11	;.loc	4 78 9                          ; ../../..//include/ops/warp/memory/util/util.cuh:78:9
	s_cselect_b32 s4, s4, s10
	s_lshl_b32 s10, s2, 1	;.loc	2 89 47                         ; 256_256_64_32_with16x32.cpp:89:47
	s_lshl_b32 s2, s18, 17
	s_or_b32 s13, s2, -2.0
	s_mov_b32 s12, s3	;.loc	2 0 47 is_stmt 0                ; 256_256_64_32_with16x32.cpp:0:47
	s_and_b32 s11, s18, 0x1fff	;.loc	4 78 9 is_stmt 1                ; ../../..//include/ops/warp/memory/util/util.cuh:78:9
	s_or_b64 s[2:3], s[12:13], s[8:9]
	s_cmp_eq_u32 s11, 0
	s_cselect_b32 s9, s9, s3
	s_cselect_b32 s8, s8, s2
	v_add_u32_e32 v001, s20, v003	;.loc	2 93 78                         ; 256_256_64_32_with16x32.cpp:93:78
	s_add_u32 s22, s20, 0x4000	;.loc	2 94 107                        ; 256_256_64_32_with16x32.cpp:94:107
	v_readfirstlane_b32 s29, v001	;.loc	2 93 25                         ; 256_256_64_32_with16x32.cpp:93:25
	v_add_u32_e32 v001, s22, v003	;.loc	2 94 78                         ; 256_256_64_32_with16x32.cpp:94:78
	s_add_u32 s24, s21, 0x4000	;.loc	2 99 107                        ; 256_256_64_32_with16x32.cpp:99:107
	v_readfirstlane_b32 s30, v001	;.loc	2 94 25                         ; 256_256_64_32_with16x32.cpp:94:25
	v_add_u32_e32 v001, s21, v003	;.loc	2 98 78                         ; 256_256_64_32_with16x32.cpp:98:78
	v_bfe_u32 v005, v000, 2, 4
	v_readfirstlane_b32 s31, v001	;.loc	2 98 25 is_stmt 0               ; 256_256_64_32_with16x32.cpp:98:25
	v_add_u32_e32 v001, s24, v003	;.loc	2 99 78 is_stmt 1               ; 256_256_64_32_with16x32.cpp:99:78
	v_bitop3_b32 v004, v000, v004, 32 bitop3:0x6c
	v_readfirstlane_b32 s33, v001	;.loc	2 99 25 is_stmt 0               ; 256_256_64_32_with16x32.cpp:99:25
	v_lshrrev_b32_e32 v001, 3, v000
	v_and_or_b32 v001, v001, 48, v005	;.loc	15 134 10 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:134:10
	v_lshrrev_b32_e32 v005, 1, v000
	v_lshrrev_b32_e32 v004, 1, v004
	v_and_b32_e32 v005, 32, v005
	v_and_or_b32 v004, v004, 24, v005
	v_mad_u64_u32 v006 v007, s[2:3], v001, s16, v004 v005	;.loc	15 147 88                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:88
	s_lshl_b32 s2, s16, 6	;.loc	15 147 75 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:75
	s_lshl_b32 s19, s15, 8	;.loc	2 116 40 is_stmt 1              ; 256_256_64_32_with16x32.cpp:116:40
	v_add_lshl_u32 v132, v006, s2, 1	;.loc	15 147 111                      ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:111
	v_mad_u64_u32 v004 v005, s[2:3], v001, s18, v004 v005	;.loc	15 147 88 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:88
	s_mul_i32 s34, s19, s18	;.loc	17 72 64 is_stmt 1              ; ../../..//include/types/global/gl.cuh:72:64
	s_lshl_b32 s2, s18, 6	;.loc	15 147 75                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:75
	s_lshl_b32 s15, s34, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	v_add_lshl_u32 v137, v004, s2, 1	;.loc	15 147 111                      ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:111
	s_mov_b32 s2, s15	;.loc	15 253 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:253:5
	s_mov_b32 s3, s31	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_mov_b32 s7, 0x110000
	s_lshl_b32 s39, s17, 8	;.loc	2 117 40                        ; 256_256_64_32_with16x32.cpp:117:40
	s_mov_b32 s11, s7	;.loc	4 78 9                          ; ../../..//include/ops/warp/memory/util/util.cuh:78:9
	v_lshlrev_b32_e32 v136, 1, v004	;.loc	15 147 111                      ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:111
	s_mov_b32 s12, s3	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_mov_b32 m0, 0	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_addk_i32 s3, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_mul_i32 s27, s39, s16	;.loc	17 72 64                        ; ../../..//include/types/global/gl.cuh:72:64
	s_lshl_b32 s13, s27, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 m0, s12	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v136, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	v_lshlrev_b32_e32 v133, 1, v006	;.loc	15 147 111                      ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:147:111
	s_mov_b32 m0, s3	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v137, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_mov_b32 s2, s13	;.loc	15 253 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:253:5
	s_mov_b32 s3, s29	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_mov_b32 s12, s3	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s3, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_add_u32 s25, s20, 0x8000	;.loc	2 95 107                        ; 256_256_64_32_with16x32.cpp:95:107
	s_mov_b32 m0, s12	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v133, s[4:7], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	v_add_u32_e32 v004, s25, v003	;.loc	2 95 78                         ; 256_256_64_32_with16x32.cpp:95:78
	s_mov_b32 m0, s3	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v132, s[4:7], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_lshl_b32 s2, s18, 7	;.loc	17 72 64                        ; ../../..//include/types/global/gl.cuh:72:64
	s_add_i32 s35, s34, s2
	s_lshl_b32 s12, s35, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s2, s12	;.loc	15 253 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:253:5
	s_mov_b32 s3, s33	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_mov_b32 s18, s3	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s3, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	v_readfirstlane_b32 s36, v004	;.loc	2 95 25                         ; 256_256_64_32_with16x32.cpp:95:25
	s_mov_b32 m0, s18	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v136, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_add_u32 s18, s20, 0xc000	;.loc	2 96 107                        ; 256_256_64_32_with16x32.cpp:96:107
	s_mov_b32 m0, s3	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v137, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_lshl_b32 s2, s16, 7	;.loc	17 72 64                        ; ../../..//include/types/global/gl.cuh:72:64
	s_add_i32 s27, s27, s2
	s_lshl_b32 s2, s27, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s3, s30	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_mov_b32 s11, s3	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s3, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	v_add_u32_e32 v004, s18, v003	;.loc	2 96 78                         ; 256_256_64_32_with16x32.cpp:96:78
	s_mov_b32 m0, s11	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v133, s[4:7], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_add_u32 s26, s21, 0x8000	;.loc	2 100 107                       ; 256_256_64_32_with16x32.cpp:100:107
	s_mov_b32 m0, s3	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v132, s[4:7], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_add_u32 s23, s21, 0xc000	;.loc	2 101 107                       ; 256_256_64_32_with16x32.cpp:101:107
	v_lshrrev_b32_e32 v001, 8, v000	;.loc	2 77 34                         ; 256_256_64_32_with16x32.cpp:77:34
	v_readfirstlane_b32 s28, v004	;.loc	2 96 25                         ; 256_256_64_32_with16x32.cpp:96:25
	v_add_u32_e32 v004, s26, v003	;.loc	2 100 78                        ; 256_256_64_32_with16x32.cpp:100:78
	v_add_u32_e32 v003, s23, v003	;.loc	2 101 78                        ; 256_256_64_32_with16x32.cpp:101:78
	v_readfirstlane_b32 s37, v004	;.loc	2 100 25                        ; 256_256_64_32_with16x32.cpp:100:25
	v_readfirstlane_b32 s38, v003	;.loc	2 101 25                        ; 256_256_64_32_with16x32.cpp:101:25
	v_cmp_eq_u32_e32 vcc, 1, v001	;.loc	2 121 18                        ; 256_256_64_32_with16x32.cpp:121:18
	s_and_saveexec_b64 s[2:3], vcc
.JUMP.LBB0_4:
	s_cbranch_execz .LBB0_4
	s_barrier	;.loc	2 122 9                         ; 256_256_64_32_with16x32.cpp:122:9
.LBB0_4:
	s_or_b64 exec, exec, s[2:3]	;.loc	2 0 9 is_stmt 0                 ; 256_256_64_32_with16x32.cpp:0:9
	s_or_b32 s2, s15, 0x80	;.loc	15 280 41 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s3, s37	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_waitcnt vmcnt(4)	;.loc	2 125 5                         ; 256_256_64_32_with16x32.cpp:125:5
	s_barrier	;.loc	2 126 5                         ; 256_256_64_32_with16x32.cpp:126:5
	s_mov_b32 s11, s3	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s3, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_mov_b32 m0, s11	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 s11, s7	;.loc	4 78 9                          ; ../../..//include/ops/warp/memory/util/util.cuh:78:9
	buffer_load_dwordx4 v136, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	v_and_b32_e32 v131, 3, v002	;.loc	2 78 34                         ; 256_256_64_32_with16x32.cpp:78:34
	s_mov_b32 m0, s3	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v137, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_or_b32 s2, s13, 0x80	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s3, s36	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_mov_b32 s13, s3	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s3, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	v_lshlrev_b32_e32 v002, 11, v131	;.loc	15 0 9 is_stmt 0                ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:0:9
	s_mov_b32 m0, s13	;.loc	15 299 9 is_stmt 1              ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v133, s[4:7], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	v_and_b32_e32 v138, 48, v000	;.loc	15 0 9 is_stmt 0                ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:0:9
	s_mov_b32 m0, s3	;.loc	15 299 9 is_stmt 1              ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v132, s[4:7], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_or_b32 s2, s12, 0x80	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s3, s38	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_mov_b32 s7, s3	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s3, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	v_lshlrev_b32_e32 v139, 6, v000	;.loc	15 0 9 is_stmt 0                ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:0:9
	s_mov_b32 m0, s7	;.loc	15 299 9 is_stmt 1              ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v136, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	v_lshlrev_b32_e32 v140, 2, v000	;.loc	15 0 9 is_stmt 0                ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:0:9
	s_mov_b32 m0, s3	;.loc	15 299 9 is_stmt 1              ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v137, s[8:11], s2 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_ashr_i32 s2, s14, 31	;.loc	2 79 29                         ; 256_256_64_32_with16x32.cpp:79:29
	s_lshr_b32 s2, s2, 26
	s_add_i32 s7, s14, s2
	s_cmpk_gt_i32 s14, 0xbf	;.loc	2 136 29                        ; 256_256_64_32_with16x32.cpp:136:29
	v_lshlrev_b32_e32 v130, 12, v001
	s_waitcnt vmcnt(6)	;.loc	2 132 5                         ; 256_256_64_32_with16x32.cpp:132:5
	s_barrier	;.loc	2 133 5                         ; 256_256_64_32_with16x32.cpp:133:5
.JUMP.LBB0_6:
	s_cbranch_scc1 .LBB0_6	;.loc	2 136 5                         ; 256_256_64_32_with16x32.cpp:136:5
	v_and_b32_e32 v003, 0x3c0, v139	;.loc	11 134 42                       ; ../../..//include/types/shared/st_shape.cuh:134:42
	v_and_b32_e32 v004, 32, v140	;.loc	11 137 56                       ; ../../..//include/types/shared/st_shape.cuh:137:56
	v_bitop3_b32 v134, v003, v004, v138 bitop3:0x36	;.loc	11 138 48                       ; ../../..//include/types/shared/st_shape.cuh:138:48
	s_mov_b64 s[14:15], 0
	s_branch .LBB0_7
.LBB0_6:
	s_mov_b64 s[14:15], -1	;.loc	11 0 48 is_stmt 0               ; ../../..//include/types/shared/st_shape.cuh:0:48
.LBB0_7:                                ; %Flow3203
	s_load_dwordx2 s[2:3], s[0:1], 0x60
	s_load_dwordx2 s[12:13], s[0:1], 0x80
	v_mov_b32_e32 v005, 0
	s_ashr_i32 s0, s7, 6
	s_andn2_b64 vcc, exec, s[14:15]
	v_lshlrev_b32_e32 v135, 1, v002
	v_mov_b32_e32 v004, v005
	v_mov_b32_e32 v003, v005
	v_mov_b32_e32 v002, v005
	v_mov_b32_e32 v009, v005
	v_mov_b32_e32 v008, v005
	v_mov_b32_e32 v007, v005
	v_mov_b32_e32 v006, v005
	v_mov_b32_e32 v013, v005
	v_mov_b32_e32 v012, v005
	v_mov_b32_e32 v011, v005
	v_mov_b32_e32 v010, v005
	v_mov_b32_e32 v017, v005
	v_mov_b32_e32 v016, v005
	v_mov_b32_e32 v015, v005
	v_mov_b32_e32 v014, v005
	v_mov_b32_e32 v021, v005
	v_mov_b32_e32 v020, v005
	v_mov_b32_e32 v019, v005
	v_mov_b32_e32 v018, v005
	v_mov_b32_e32 v025, v005
	v_mov_b32_e32 v024, v005
	v_mov_b32_e32 v023, v005
	v_mov_b32_e32 v022, v005
	v_mov_b32_e32 v029, v005
	v_mov_b32_e32 v028, v005
	v_mov_b32_e32 v027, v005
	v_mov_b32_e32 v026, v005
	v_mov_b32_e32 v033, v005
	v_mov_b32_e32 v032, v005
	v_mov_b32_e32 v031, v005
	v_mov_b32_e32 v030, v005
	v_mov_b32_e32 v037, v005
	v_mov_b32_e32 v036, v005
	v_mov_b32_e32 v035, v005
	v_mov_b32_e32 v034, v005
	v_mov_b32_e32 v041, v005
	v_mov_b32_e32 v040, v005
	v_mov_b32_e32 v039, v005
	v_mov_b32_e32 v038, v005
	v_mov_b32_e32 v045, v005
	v_mov_b32_e32 v044, v005
	v_mov_b32_e32 v043, v005
	v_mov_b32_e32 v042, v005
	v_mov_b32_e32 v049, v005
	v_mov_b32_e32 v048, v005
	v_mov_b32_e32 v047, v005
	v_mov_b32_e32 v046, v005
	v_mov_b32_e32 v053, v005
	v_mov_b32_e32 v052, v005
	v_mov_b32_e32 v051, v005
	v_mov_b32_e32 v050, v005
	v_mov_b32_e32 v057, v005
	v_mov_b32_e32 v056, v005
	v_mov_b32_e32 v055, v005
	v_mov_b32_e32 v054, v005
	v_mov_b32_e32 v061, v005
	v_mov_b32_e32 v060, v005
	v_mov_b32_e32 v059, v005
	v_mov_b32_e32 v058, v005
	v_mov_b32_e32 v065, v005
	v_mov_b32_e32 v064, v005
	v_mov_b32_e32 v063, v005
	v_mov_b32_e32 v062, v005
	v_mov_b32_e32 v069, v005
	v_mov_b32_e32 v068, v005
	v_mov_b32_e32 v067, v005
	v_mov_b32_e32 v066, v005
	v_mov_b32_e32 v073, v005
	v_mov_b32_e32 v072, v005
	v_mov_b32_e32 v071, v005
	v_mov_b32_e32 v070, v005
	v_mov_b32_e32 v077, v005
	v_mov_b32_e32 v076, v005
	v_mov_b32_e32 v075, v005
	v_mov_b32_e32 v074, v005
	v_mov_b32_e32 v081, v005
	v_mov_b32_e32 v080, v005
	v_mov_b32_e32 v079, v005
	v_mov_b32_e32 v078, v005
	v_mov_b32_e32 v085, v005
	v_mov_b32_e32 v084, v005
	v_mov_b32_e32 v083, v005
	v_mov_b32_e32 v082, v005
	v_mov_b32_e32 v089, v005
	v_mov_b32_e32 v088, v005
	v_mov_b32_e32 v087, v005
	v_mov_b32_e32 v086, v005
	v_mov_b32_e32 v093, v005
	v_mov_b32_e32 v092, v005
	v_mov_b32_e32 v091, v005
	v_mov_b32_e32 v090, v005
	v_mov_b32_e32 v097, v005
	v_mov_b32_e32 v096, v005
	v_mov_b32_e32 v095, v005
	v_mov_b32_e32 v094, v005
	v_mov_b32_e32 v101, v005
	v_mov_b32_e32 v100, v005
	v_mov_b32_e32 v099, v005
	v_mov_b32_e32 v098, v005
	v_mov_b32_e32 v105, v005
	v_mov_b32_e32 v104, v005
	v_mov_b32_e32 v103, v005
	v_mov_b32_e32 v102, v005
	v_mov_b32_e32 v109, v005
	v_mov_b32_e32 v108, v005
	v_mov_b32_e32 v107, v005
	v_mov_b32_e32 v106, v005
	v_mov_b32_e32 v113, v005
	v_mov_b32_e32 v112, v005
	v_mov_b32_e32 v111, v005
	v_mov_b32_e32 v110, v005
	v_mov_b32_e32 v117, v005
	v_mov_b32_e32 v116, v005
	v_mov_b32_e32 v115, v005
	v_mov_b32_e32 v114, v005
	v_mov_b32_e32 v121, v005
	v_mov_b32_e32 v120, v005
	v_mov_b32_e32 v119, v005
	v_mov_b32_e32 v118, v005
	v_mov_b32_e32 v125, v005
	v_mov_b32_e32 v124, v005
	v_mov_b32_e32 v123, v005
	v_mov_b32_e32 v122, v005
	v_mov_b32_e32 v129, v005
	v_mov_b32_e32 v128, v005
	v_mov_b32_e32 v127, v005
	v_mov_b32_e32 v126, v005
.JUMP.LBB0_10:
	s_cbranch_vccnz .LBB0_10
	v_and_b32_e32 v002, 0x3c0, v139
	v_and_b32_e32 v003, 32, v140
	s_mul_i32 s7, s17, s16	;.loc	2 136 5 is_stmt 1               ; 256_256_64_32_with16x32.cpp:136:5
	v_bitop3_b32 v134, v002, v003, v138 bitop3:0x36
	v_mov_b32_e32 v002, 0
	v_lshlrev_b32_e32 v003, 13, v001
	s_waitcnt lgkmcnt(0)
	s_add_i32 s13, s39, 0x80
	s_lshl_b32 s14, s7, 8
	s_mov_b32 s7, 0x110000
	s_add_i32 s1, s0, -2
	v_add3_u32 v138, s21, v135, v134
	v_add3_u32 v139, s20, v003, v134
	v_add3_u32 v140, s24, v135, v134
	v_add3_u32 v141, s22, v003, v134
	v_add3_u32 v142, s26, v135, v134
	v_add3_u32 v143, s25, v003, v134
	v_add3_u32 v144, s23, v135, v134
	v_add3_u32 v145, s18, v003, v134
	s_mul_i32 s13, s13, s16
	s_mov_b32 s15, 0
	s_mov_b32 s11, s7
	s_mov_b32 s16, 0
	v_mov_b32_e32 v003, v002
	v_mov_b32_e32 v004, v002
	v_mov_b32_e32 v005, v002
	v_mov_b32_e32 v006, v002
	v_mov_b32_e32 v007, v002
	v_mov_b32_e32 v008, v002
	v_mov_b32_e32 v009, v002
	v_mov_b32_e32 v010, v002
	v_mov_b32_e32 v011, v002
	v_mov_b32_e32 v012, v002
	v_mov_b32_e32 v013, v002
	v_mov_b32_e32 v014, v002
	v_mov_b32_e32 v015, v002
	v_mov_b32_e32 v016, v002
	v_mov_b32_e32 v017, v002
	v_mov_b32_e32 v018, v002
	v_mov_b32_e32 v019, v002
	v_mov_b32_e32 v020, v002
	v_mov_b32_e32 v021, v002
	v_mov_b32_e32 v022, v002
	v_mov_b32_e32 v023, v002
	v_mov_b32_e32 v024, v002
	v_mov_b32_e32 v025, v002
	v_mov_b32_e32 v026, v002
	v_mov_b32_e32 v027, v002
	v_mov_b32_e32 v028, v002
	v_mov_b32_e32 v029, v002
	v_mov_b32_e32 v030, v002
	v_mov_b32_e32 v031, v002
	v_mov_b32_e32 v032, v002
	v_mov_b32_e32 v033, v002
	v_mov_b32_e32 v034, v002
	v_mov_b32_e32 v035, v002
	v_mov_b32_e32 v036, v002
	v_mov_b32_e32 v037, v002
	v_mov_b32_e32 v038, v002
	v_mov_b32_e32 v039, v002
	v_mov_b32_e32 v040, v002
	v_mov_b32_e32 v041, v002
	v_mov_b32_e32 v042, v002
	v_mov_b32_e32 v043, v002
	v_mov_b32_e32 v044, v002
	v_mov_b32_e32 v045, v002
	v_mov_b32_e32 v046, v002
	v_mov_b32_e32 v047, v002
	v_mov_b32_e32 v048, v002
	v_mov_b32_e32 v049, v002
	v_mov_b32_e32 v050, v002
	v_mov_b32_e32 v051, v002
	v_mov_b32_e32 v052, v002
	v_mov_b32_e32 v053, v002
	v_mov_b32_e32 v054, v002
	v_mov_b32_e32 v055, v002
	v_mov_b32_e32 v056, v002
	v_mov_b32_e32 v057, v002
	v_mov_b32_e32 v058, v002
	v_mov_b32_e32 v059, v002
	v_mov_b32_e32 v060, v002
	v_mov_b32_e32 v061, v002
	v_mov_b32_e32 v062, v002
	v_mov_b32_e32 v063, v002
	v_mov_b32_e32 v064, v002
	v_mov_b32_e32 v065, v002
	v_mov_b32_e32 v066, v002
	v_mov_b32_e32 v067, v002
	v_mov_b32_e32 v068, v002
	v_mov_b32_e32 v069, v002
	v_mov_b32_e32 v070, v002
	v_mov_b32_e32 v071, v002
	v_mov_b32_e32 v072, v002
	v_mov_b32_e32 v073, v002
	v_mov_b32_e32 v074, v002
	v_mov_b32_e32 v075, v002
	v_mov_b32_e32 v076, v002
	v_mov_b32_e32 v077, v002
	v_mov_b32_e32 v078, v002
	v_mov_b32_e32 v079, v002
	v_mov_b32_e32 v080, v002
	v_mov_b32_e32 v081, v002
	v_mov_b32_e32 v082, v002
	v_mov_b32_e32 v083, v002
	v_mov_b32_e32 v084, v002
	v_mov_b32_e32 v085, v002
	v_mov_b32_e32 v086, v002
	v_mov_b32_e32 v087, v002
	v_mov_b32_e32 v088, v002
	v_mov_b32_e32 v089, v002
	v_mov_b32_e32 v090, v002
	v_mov_b32_e32 v091, v002
	v_mov_b32_e32 v092, v002
	v_mov_b32_e32 v093, v002
	v_mov_b32_e32 v094, v002
	v_mov_b32_e32 v095, v002
	v_mov_b32_e32 v096, v002
	v_mov_b32_e32 v097, v002
	v_mov_b32_e32 v098, v002
	v_mov_b32_e32 v099, v002
	v_mov_b32_e32 v100, v002
	v_mov_b32_e32 v101, v002
	v_mov_b32_e32 v102, v002
	v_mov_b32_e32 v103, v002
	v_mov_b32_e32 v104, v002
	v_mov_b32_e32 v105, v002
	v_mov_b32_e32 v106, v002
	v_mov_b32_e32 v107, v002
	v_mov_b32_e32 v108, v002
	v_mov_b32_e32 v109, v002
	v_mov_b32_e32 v110, v002
	v_mov_b32_e32 v111, v002
	v_mov_b32_e32 v112, v002
	v_mov_b32_e32 v113, v002
	v_mov_b32_e32 v114, v002
	v_mov_b32_e32 v115, v002
	v_mov_b32_e32 v116, v002
	v_mov_b32_e32 v117, v002
	v_mov_b32_e32 v118, v002
	v_mov_b32_e32 v119, v002
	v_mov_b32_e32 v120, v002
	v_mov_b32_e32 v121, v002
	v_mov_b32_e32 v122, v002
	v_mov_b32_e32 v123, v002
	v_mov_b32_e32 v124, v002
	v_mov_b32_e32 v125, v002
	v_mov_b32_e32 v126, v002
	v_mov_b32_e32 v127, v002
	v_mov_b32_e32 v128, v002
	v_mov_b32_e32 v129, v002
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	.file	26 "../../..//include/ops/warp/memory/tile" "shared_to_register.cuh"
	ds_read_b128 v146 v147 v148 v149, v138 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v150 v151 v152 v153, v138 offset:0x400
	ds_read_b128 v154 v155 v156 v157, v138 offset:0x800
	ds_read_b128 v158 v159 v160 v161, v138 offset:0xc00
	ds_read_b128 v162 v163 v164 v165, v139 offset:0	;.loc	26 78 37 is_stmt 0              ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v166 v167 v168 v169, v139 offset:0x400
	ds_read_b128 v170 v171 v172 v173, v139 offset:0x800
	ds_read_b128 v174 v175 v176 v177, v139 offset:0xc00
	ds_read_b128 v178 v179 v180 v181, v139 offset:0x1000
	s_add_i32 s39, s13, s15	;.loc	15 252 9 is_stmt 1              ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v182 v183 v184 v185, v139 offset:0x1400	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_add_i32 s40, s39, 64	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v186 v187 v188 v189, v139 offset:0x1800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s40, s40, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s41, s28	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	ds_read_b128 v190 v191 v192 v193, v139 offset:0x1c00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s42, s41	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s41, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s42
	buffer_load_dwordx4 v133, s[4:7], s40 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s41
	buffer_load_dwordx4 v132, s[4:7], s40 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_waitcnt lgkmcnt(8)	;.loc	2 143 9                         ; 256_256_64_32_with16x32.cpp:143:9
	s_barrier	;.loc	2 144 9                         ; 256_256_64_32_with16x32.cpp:144:9
	s_waitcnt lgkmcnt(0)	;.loc	2 146 9                         ; 256_256_64_32_with16x32.cpp:146:9
	s_setprio 1	;.loc	2 147 9                         ; 256_256_64_32_with16x32.cpp:147:9
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v162 v163 v164 v165, v146 v147 v148 v149, v126 v127 v128 v129	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v162 v163 v164 v165, v154 v155 v156 v157, v122 v123 v124 v125
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v170 v171 v172 v173, v146 v147 v148 v149, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v170 v171 v172 v173, v154 v155 v156 v157, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v178 v179 v180 v181, v146 v147 v148 v149, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v178 v179 v180 v181, v154 v155 v156 v157, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v186 v187 v188 v189, v146 v147 v148 v149, v102 v103 v104 v105
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v186 v187 v188 v189, v154 v155 v156 v157, v098 v099 v100 v101
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v166 v167 v168 v169, v150 v151 v152 v153, v126 v127 v128 v129	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v166 v167 v168 v169, v158 v159 v160 v161, v122 v123 v124 v125
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v174 v175 v176 v177, v150 v151 v152 v153, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v174 v175 v176 v177, v158 v159 v160 v161, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v182 v183 v184 v185, v150 v151 v152 v153, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v182 v183 v184 v185, v158 v159 v160 v161, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v190 v191 v192 v193, v150 v151 v152 v153, v102 v103 v104 v105
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v190 v191 v192 v193, v158 v159 v160 v161, v098 v099 v100 v101
	s_setprio 0	;.loc	2 149 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:149:9
	s_barrier	;.loc	2 150 9                         ; 256_256_64_32_with16x32.cpp:150:9
	ds_read_b128 v194 v195 v196 v197, v140 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_add_i32 s40, s34, s15	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v198 v199 v200 v201, v140 offset:0x400	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_add_i32 s41, s40, 0x80	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v202 v203 v204 v205, v140 offset:0x800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s41, s41, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s42, s31	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	ds_read_b128 v206 v207 v208 v209, v140 offset:0xc00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s43, s42	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s42, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_add_i32 s16, s16, 2	;.loc	2 155 51                        ; 256_256_64_32_with16x32.cpp:155:51
	s_mov_b32 m0, s43	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v136, s[8:11], s41 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s42
	buffer_load_dwordx4 v137, s[8:11], s41 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_barrier	;.loc	2 156 9                         ; 256_256_64_32_with16x32.cpp:156:9
	s_waitcnt lgkmcnt(0)	;.loc	2 158 9                         ; 256_256_64_32_with16x32.cpp:158:9
	s_setprio 1	;.loc	2 159 9                         ; 256_256_64_32_with16x32.cpp:159:9
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v162 v163 v164 v165, v194 v195 v196 v197, v094 v095 v096 v097	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v162 v163 v164 v165, v202 v203 v204 v205, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v170 v171 v172 v173, v194 v195 v196 v197, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v170 v171 v172 v173, v202 v203 v204 v205, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v178 v179 v180 v181, v194 v195 v196 v197, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v178 v179 v180 v181, v202 v203 v204 v205, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v186 v187 v188 v189, v194 v195 v196 v197, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v186 v187 v188 v189, v202 v203 v204 v205, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v166 v167 v168 v169, v198 v199 v200 v201, v094 v095 v096 v097	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v166 v167 v168 v169, v206 v207 v208 v209, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v174 v175 v176 v177, v198 v199 v200 v201, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v174 v175 v176 v177, v206 v207 v208 v209, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v182 v183 v184 v185, v198 v199 v200 v201, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v182 v183 v184 v185, v206 v207 v208 v209, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v190 v191 v192 v193, v198 v199 v200 v201, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v190 v191 v192 v193, v206 v207 v208 v209, v066 v067 v068 v069
	s_setprio 0	;.loc	2 161 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:161:9
	s_barrier	;.loc	2 162 9                         ; 256_256_64_32_with16x32.cpp:162:9
	ds_read_b128 v162 v163 v164 v165, v141 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v166 v167 v168 v169, v141 offset:0x400
	ds_read_b128 v170 v171 v172 v173, v141 offset:0x800
	ds_read_b128 v174 v175 v176 v177, v141 offset:0xc00
	ds_read_b128 v178 v179 v180 v181, v141 offset:0x1000
	s_add_i32 s41, s14, s15	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v182 v183 v184 v185, v141 offset:0x1400	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_add_i32 s42, s41, 0x80	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v186 v187 v188 v189, v141 offset:0x1800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s42, s42, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s43, s29	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	ds_read_b128 v190 v191 v192 v193, v141 offset:0x1c00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s44, s43	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s43, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s44
	buffer_load_dwordx4 v133, s[4:7], s42 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s43
	buffer_load_dwordx4 v132, s[4:7], s42 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_barrier	;.loc	2 167 9                         ; 256_256_64_32_with16x32.cpp:167:9
	s_waitcnt lgkmcnt(0)	;.loc	2 169 9                         ; 256_256_64_32_with16x32.cpp:169:9
	s_setprio 1	;.loc	2 170 9                         ; 256_256_64_32_with16x32.cpp:170:9
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v162 v163 v164 v165, v146 v147 v148 v149, v062 v063 v064 v065	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v162 v163 v164 v165, v154 v155 v156 v157, v058 v059 v060 v061
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v170 v171 v172 v173, v146 v147 v148 v149, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v170 v171 v172 v173, v154 v155 v156 v157, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v178 v179 v180 v181, v146 v147 v148 v149, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v178 v179 v180 v181, v154 v155 v156 v157, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v186 v187 v188 v189, v146 v147 v148 v149, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v186 v187 v188 v189, v154 v155 v156 v157, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v166 v167 v168 v169, v150 v151 v152 v153, v062 v063 v064 v065	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v166 v167 v168 v169, v158 v159 v160 v161, v058 v059 v060 v061
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v174 v175 v176 v177, v150 v151 v152 v153, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v174 v175 v176 v177, v158 v159 v160 v161, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v182 v183 v184 v185, v150 v151 v152 v153, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v182 v183 v184 v185, v158 v159 v160 v161, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v190 v191 v192 v193, v150 v151 v152 v153, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v190 v191 v192 v193, v158 v159 v160 v161, v034 v035 v036 v037
	s_setprio 0	;.loc	2 172 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:172:9
	s_barrier	;.loc	2 173 9                         ; 256_256_64_32_with16x32.cpp:173:9
	ds_read_b128 v146 v147 v148 v149, v142 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_add_i32 s42, s35, s15	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v150 v151 v152 v153, v142 offset:0x400	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_add_i32 s43, s42, 0x80	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v154 v155 v156 v157, v142 offset:0x800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s43, s43, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s44, s33	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	ds_read_b128 v158 v159 v160 v161, v142 offset:0xc00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s45, s44	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s44, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s45
	buffer_load_dwordx4 v136, s[8:11], s43 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s44
	buffer_load_dwordx4 v137, s[8:11], s43 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_waitcnt vmcnt(6)	;.loc	2 179 9                         ; 256_256_64_32_with16x32.cpp:179:9
	s_barrier	;.loc	2 180 9                         ; 256_256_64_32_with16x32.cpp:180:9
	s_setprio 1	;.loc	2 182 9                         ; 256_256_64_32_with16x32.cpp:182:9
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v162 v163 v164 v165, v194 v195 v196 v197, v030 v031 v032 v033	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v162 v163 v164 v165, v202 v203 v204 v205, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v170 v171 v172 v173, v194 v195 v196 v197, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v170 v171 v172 v173, v202 v203 v204 v205, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v178 v179 v180 v181, v194 v195 v196 v197, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v178 v179 v180 v181, v202 v203 v204 v205, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v186 v187 v188 v189, v194 v195 v196 v197, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v186 v187 v188 v189, v202 v203 v204 v205, v002 v003 v004 v005
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v166 v167 v168 v169, v198 v199 v200 v201, v030 v031 v032 v033	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v166 v167 v168 v169, v206 v207 v208 v209, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v174 v175 v176 v177, v198 v199 v200 v201, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v174 v175 v176 v177, v206 v207 v208 v209, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v182 v183 v184 v185, v198 v199 v200 v201, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v182 v183 v184 v185, v206 v207 v208 v209, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v190 v191 v192 v193, v198 v199 v200 v201, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v190 v191 v192 v193, v206 v207 v208 v209, v002 v003 v004 v005
	s_setprio 0	;.loc	2 184 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:184:9
	s_barrier	;.loc	2 185 9                         ; 256_256_64_32_with16x32.cpp:185:9
	ds_read_b128 v162 v163 v164 v165, v143 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v166 v167 v168 v169, v143 offset:0x400
	ds_read_b128 v170 v171 v172 v173, v143 offset:0x800
	ds_read_b128 v174 v175 v176 v177, v143 offset:0xc00
	ds_read_b128 v178 v179 v180 v181, v143 offset:0x1000
	ds_read_b128 v182 v183 v184 v185, v143 offset:0x1400
	s_addk_i32 s39, 0x80	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v186 v187 v188 v189, v143 offset:0x1800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s39, s39, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s43, s30	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	ds_read_b128 v190 v191 v192 v193, v143 offset:0x1c00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s44, s43	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s43, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s44
	buffer_load_dwordx4 v133, s[4:7], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s43
	buffer_load_dwordx4 v132, s[4:7], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_waitcnt lgkmcnt(8)	;.loc	2 191 9                         ; 256_256_64_32_with16x32.cpp:191:9
	s_barrier	;.loc	2 192 9                         ; 256_256_64_32_with16x32.cpp:192:9
	s_waitcnt lgkmcnt(0)	;.loc	2 194 9                         ; 256_256_64_32_with16x32.cpp:194:9
	s_setprio 1	;.loc	2 195 9                         ; 256_256_64_32_with16x32.cpp:195:9
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v162 v163 v164 v165, v146 v147 v148 v149, v126 v127 v128 v129	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v162 v163 v164 v165, v154 v155 v156 v157, v122 v123 v124 v125
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v170 v171 v172 v173, v146 v147 v148 v149, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v170 v171 v172 v173, v154 v155 v156 v157, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v178 v179 v180 v181, v146 v147 v148 v149, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v178 v179 v180 v181, v154 v155 v156 v157, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v186 v187 v188 v189, v146 v147 v148 v149, v102 v103 v104 v105
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v186 v187 v188 v189, v154 v155 v156 v157, v098 v099 v100 v101
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v166 v167 v168 v169, v150 v151 v152 v153, v126 v127 v128 v129	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v166 v167 v168 v169, v158 v159 v160 v161, v122 v123 v124 v125
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v174 v175 v176 v177, v150 v151 v152 v153, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v174 v175 v176 v177, v158 v159 v160 v161, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v182 v183 v184 v185, v150 v151 v152 v153, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v182 v183 v184 v185, v158 v159 v160 v161, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v190 v191 v192 v193, v150 v151 v152 v153, v102 v103 v104 v105
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v190 v191 v192 v193, v158 v159 v160 v161, v098 v099 v100 v101
	s_setprio 0	;.loc	2 197 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:197:9
	s_barrier	;.loc	2 198 9                         ; 256_256_64_32_with16x32.cpp:198:9
	ds_read_b128 v194 v195 v196 v197, v144 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v198 v199 v200 v201, v144 offset:0x400
	s_addk_i32 s40, 0xc0	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v202 v203 v204 v205, v144 offset:0x800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s39, s40, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s40, s37	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	ds_read_b128 v206 v207 v208 v209, v144 offset:0xc00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s43, s40	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s40, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s43
	buffer_load_dwordx4 v136, s[8:11], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s40
	buffer_load_dwordx4 v137, s[8:11], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_barrier	;.loc	2 204 9                         ; 256_256_64_32_with16x32.cpp:204:9
	s_waitcnt lgkmcnt(0)	;.loc	2 206 9                         ; 256_256_64_32_with16x32.cpp:206:9
	s_setprio 1	;.loc	2 207 9                         ; 256_256_64_32_with16x32.cpp:207:9
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v162 v163 v164 v165, v194 v195 v196 v197, v094 v095 v096 v097	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v162 v163 v164 v165, v202 v203 v204 v205, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v170 v171 v172 v173, v194 v195 v196 v197, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v170 v171 v172 v173, v202 v203 v204 v205, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v178 v179 v180 v181, v194 v195 v196 v197, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v178 v179 v180 v181, v202 v203 v204 v205, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v186 v187 v188 v189, v194 v195 v196 v197, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v186 v187 v188 v189, v202 v203 v204 v205, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v166 v167 v168 v169, v198 v199 v200 v201, v094 v095 v096 v097	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v166 v167 v168 v169, v206 v207 v208 v209, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v174 v175 v176 v177, v198 v199 v200 v201, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v174 v175 v176 v177, v206 v207 v208 v209, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v182 v183 v184 v185, v198 v199 v200 v201, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v182 v183 v184 v185, v206 v207 v208 v209, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v190 v191 v192 v193, v198 v199 v200 v201, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v190 v191 v192 v193, v206 v207 v208 v209, v066 v067 v068 v069
	s_setprio 0	;.loc	2 209 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:209:9
	s_barrier	;.loc	2 210 9                         ; 256_256_64_32_with16x32.cpp:210:9
	ds_read_b128 v162 v163 v164 v165, v145 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v166 v167 v168 v169, v145 offset:0x400
	ds_read_b128 v170 v171 v172 v173, v145 offset:0x800
	ds_read_b128 v174 v175 v176 v177, v145 offset:0xc00
	ds_read_b128 v178 v179 v180 v181, v145 offset:0x1000
	ds_read_b128 v182 v183 v184 v185, v145 offset:0x1400
	s_addk_i32 s41, 0xc0	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	ds_read_b128 v186 v187 v188 v189, v145 offset:0x1800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s39, s41, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s40, s36	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	ds_read_b128 v190 v191 v192 v193, v145 offset:0x1c00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s41, s40	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s40, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s41
	buffer_load_dwordx4 v133, s[4:7], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s40
	buffer_load_dwordx4 v132, s[4:7], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_barrier	;.loc	2 215 9                         ; 256_256_64_32_with16x32.cpp:215:9
	s_waitcnt lgkmcnt(0)	;.loc	2 217 9                         ; 256_256_64_32_with16x32.cpp:217:9
	s_setprio 1	;.loc	2 218 9                         ; 256_256_64_32_with16x32.cpp:218:9
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v162 v163 v164 v165, v146 v147 v148 v149, v062 v063 v064 v065	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v162 v163 v164 v165, v154 v155 v156 v157, v058 v059 v060 v061
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v170 v171 v172 v173, v146 v147 v148 v149, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v170 v171 v172 v173, v154 v155 v156 v157, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v178 v179 v180 v181, v146 v147 v148 v149, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v178 v179 v180 v181, v154 v155 v156 v157, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v186 v187 v188 v189, v146 v147 v148 v149, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v186 v187 v188 v189, v154 v155 v156 v157, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v166 v167 v168 v169, v150 v151 v152 v153, v062 v063 v064 v065	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v166 v167 v168 v169, v158 v159 v160 v161, v058 v059 v060 v061
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v174 v175 v176 v177, v150 v151 v152 v153, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v174 v175 v176 v177, v158 v159 v160 v161, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v182 v183 v184 v185, v150 v151 v152 v153, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v182 v183 v184 v185, v158 v159 v160 v161, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v190 v191 v192 v193, v150 v151 v152 v153, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v190 v191 v192 v193, v158 v159 v160 v161, v034 v035 v036 v037
	s_setprio 0	;.loc	2 220 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:220:9
	s_barrier	;.loc	2 221 9                         ; 256_256_64_32_with16x32.cpp:221:9
	s_addk_i32 s42, 0xc0	;.loc	15 252 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:252:9
	s_lshl_b32 s39, s42, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	s_mov_b32 s40, s38	;.loc	15 292 5                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:292:5
	s_mov_b32 s41, s40	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_addk_i32 s40, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s41
	buffer_load_dwordx4 v136, s[8:11], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_nop 0	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, s40
	buffer_load_dwordx4 v137, s[8:11], s39 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_waitcnt vmcnt(6)	;.loc	2 225 9                         ; 256_256_64_32_with16x32.cpp:225:9
	s_barrier	;.loc	2 226 9                         ; 256_256_64_32_with16x32.cpp:226:9
	s_setprio 1	;.loc	2 228 9                         ; 256_256_64_32_with16x32.cpp:228:9
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v162 v163 v164 v165, v194 v195 v196 v197, v030 v031 v032 v033	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v162 v163 v164 v165, v202 v203 v204 v205, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v170 v171 v172 v173, v194 v195 v196 v197, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v170 v171 v172 v173, v202 v203 v204 v205, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v178 v179 v180 v181, v194 v195 v196 v197, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v178 v179 v180 v181, v202 v203 v204 v205, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v186 v187 v188 v189, v194 v195 v196 v197, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v186 v187 v188 v189, v202 v203 v204 v205, v002 v003 v004 v005
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v166 v167 v168 v169, v198 v199 v200 v201, v030 v031 v032 v033	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v166 v167 v168 v169, v206 v207 v208 v209, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v174 v175 v176 v177, v198 v199 v200 v201, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v174 v175 v176 v177, v206 v207 v208 v209, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v182 v183 v184 v185, v198 v199 v200 v201, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v182 v183 v184 v185, v206 v207 v208 v209, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v190 v191 v192 v193, v198 v199 v200 v201, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v190 v191 v192 v193, v206 v207 v208 v209, v002 v003 v004 v005
	s_setprio 0	;.loc	2 230 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:230:9
	s_addk_i32 s15, 0x80	;.loc	2 136 29                        ; 256_256_64_32_with16x32.cpp:136:29
	s_cmp_ge_i32 s16, s1
	s_barrier	;.loc	2 231 9                         ; 256_256_64_32_with16x32.cpp:231:9
.JUMP.LBB0_9:
	s_cbranch_scc0 .LBB0_9	;.loc	2 136 5                         ; 256_256_64_32_with16x32.cpp:136:5
.LBB0_10:                               ; %Flow3204
	v_add3_u32 v137, s21, v135, v134	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	ds_read_b128 v138 v139 v140 v141, v137 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v142 v143 v144 v145, v137 offset:0x400
	ds_read_b128 v146 v147 v148 v149, v137 offset:0x800
	ds_read_b128 v150 v151 v152 v153, v137 offset:0xc00
	v_lshlrev_b32_e32 v130, 1, v130	;.loc	12 149 17                       ; ../../..//include/types/shared/st.cuh:149:17
	v_add3_u32 v137, s20, v130, v134	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	ds_read_b128 v154 v155 v156 v157, v137 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v158 v159 v160 v161, v137 offset:0x400
	ds_read_b128 v162 v163 v164 v165, v137 offset:0x800
	ds_read_b128 v166 v167 v168 v169, v137 offset:0xc00
	s_lshl_b32 s0, s0, 6	;.loc	23 61 18                        ; ../../..//include/types/global/util.cuh:61:18
	ds_read_b128 v170 v171 v172 v173, v137 offset:0x1000	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_add_i32 s0, s0, s27	;.loc	23 61 18                        ; ../../..//include/types/global/util.cuh:61:18
	ds_read_b128 v174 v175 v176 v177, v137 offset:0x1400	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_sub_i32 s0, s0, 64	;.loc	17 72 72                        ; ../../..//include/types/global/gl.cuh:72:72
	ds_read_b128 v178 v179 v180 v181, v137 offset:0x1800	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_lshl_b32 s0, s0, 1	;.loc	15 280 41                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:280:41
	ds_read_b128 v182 v183 v184 v185, v137 offset:0x1c00	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	s_mov_b32 s1, s28	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_mov_b32 s7, 0x110000	;.loc	4 78 9                          ; ../../..//include/ops/warp/memory/util/util.cuh:78:9
	s_mov_b32 m0, s1	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	s_mov_b32 m0, 0
	s_add_i32 s1, s28, 0x2000	;.loc	15 311 17                       ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:311:17
	buffer_load_dwordx4 v133, s[4:7], s0 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	v_lshrrev_b32_e32 v136, 2, v000	;.loc	15 297 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:297:9
	s_mov_b32 m0, s1	;.loc	15 299 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:299:9
	buffer_load_dwordx4 v132, s[4:7], s0 offen lds	;.loc	15 300 9                        ; ../../..//include/ops/warp/memory/tile/global_to_shared.cuh:300:9
	s_barrier	;.loc	2 242 9                         ; 256_256_64_32_with16x32.cpp:242:9
	s_waitcnt lgkmcnt(0)	;.loc	2 243 9                         ; 256_256_64_32_with16x32.cpp:243:9
	s_setprio 1	;.loc	2 245 9                         ; 256_256_64_32_with16x32.cpp:245:9
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v154 v155 v156 v157, v138 v139 v140 v141, v126 v127 v128 v129	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v154 v155 v156 v157, v146 v147 v148 v149, v122 v123 v124 v125
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v162 v163 v164 v165, v138 v139 v140 v141, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v162 v163 v164 v165, v146 v147 v148 v149, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v170 v171 v172 v173, v138 v139 v140 v141, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v170 v171 v172 v173, v146 v147 v148 v149, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v178 v179 v180 v181, v138 v139 v140 v141, v102 v103 v104 v105
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v178 v179 v180 v181, v146 v147 v148 v149, v098 v099 v100 v101
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v158 v159 v160 v161, v142 v143 v144 v145, v126 v127 v128 v129	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v158 v159 v160 v161, v150 v151 v152 v153, v122 v123 v124 v125
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v166 v167 v168 v169, v142 v143 v144 v145, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v166 v167 v168 v169, v150 v151 v152 v153, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v174 v175 v176 v177, v142 v143 v144 v145, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v174 v175 v176 v177, v150 v151 v152 v153, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v182 v183 v184 v185, v142 v143 v144 v145, v102 v103 v104 v105
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v182 v183 v184 v185, v150 v151 v152 v153, v098 v099 v100 v101
	s_setprio 0	;.loc	2 247 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:247:9
	s_barrier	;.loc	2 248 9                         ; 256_256_64_32_with16x32.cpp:248:9
	v_add3_u32 v132, s24, v135, v134	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	ds_read_b128 v186 v187 v188 v189, v132 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v190 v191 v192 v193, v132 offset:0x400
	ds_read_b128 v194 v195 v196 v197, v132 offset:0x800
	ds_read_b128 v198 v199 v200 v201, v132 offset:0xc00
	s_barrier	;.loc	2 252 9                         ; 256_256_64_32_with16x32.cpp:252:9
	s_waitcnt lgkmcnt(0)	;.loc	2 254 9                         ; 256_256_64_32_with16x32.cpp:254:9
	s_setprio 1	;.loc	2 255 9                         ; 256_256_64_32_with16x32.cpp:255:9
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v154 v155 v156 v157, v194 v195 v196 v197, v090 v091 v092 v093	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v162 v163 v164 v165, v194 v195 v196 v197, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v170 v171 v172 v173, v186 v187 v188 v189, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v170 v171 v172 v173, v194 v195 v196 v197, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v178 v179 v180 v181, v186 v187 v188 v189, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v178 v179 v180 v181, v194 v195 v196 v197, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v154 v155 v156 v157, v186 v187 v188 v189, v094 v095 v096 v097
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v158 v159 v160 v161, v198 v199 v200 v201, v090 v091 v092 v093	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v162 v163 v164 v165, v186 v187 v188 v189, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v166 v167 v168 v169, v198 v199 v200 v201, v082 v083 v084 v085	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v174 v175 v176 v177, v190 v191 v192 v193, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v174 v175 v176 v177, v198 v199 v200 v201, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v182 v183 v184 v185, v190 v191 v192 v193, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v182 v183 v184 v185, v198 v199 v200 v201, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v202 v203 v204 v205, v158 v159 v160 v161, v190 v191 v192 v193, v094 v095 v096 v097
	v_mfma_f32_16x16x32_bf16 v154 v155 v156 v157, v166 v167 v168 v169, v190 v191 v192 v193, v086 v087 v088 v089
	s_setprio 0	;.loc	2 257 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:257:9
	s_barrier	;.loc	2 258 9                         ; 256_256_64_32_with16x32.cpp:258:9
	v_add3_u32 v132, s22, v130, v134	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	ds_read_b128 v086 v087 v088 v089, v132 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v094 v095 v096 v097, v132 offset:0x400
	ds_read_b128 v158 v159 v160 v161, v132 offset:0x800
	ds_read_b128 v162 v163 v164 v165, v132 offset:0xc00
	ds_read_b128 v166 v167 v168 v169, v132 offset:0x1000
	ds_read_b128 v170 v171 v172 v173, v132 offset:0x1400
	ds_read_b128 v174 v175 v176 v177, v132 offset:0x1800
	ds_read_b128 v178 v179 v180 v181, v132 offset:0x1c00
	s_waitcnt vmcnt(4)	;.loc	2 262 9                         ; 256_256_64_32_with16x32.cpp:262:9
	s_barrier	;.loc	2 263 9                         ; 256_256_64_32_with16x32.cpp:263:9
	s_waitcnt lgkmcnt(0)	;.loc	2 265 9                         ; 256_256_64_32_with16x32.cpp:265:9
	s_setprio 1	;.loc	2 266 9                         ; 256_256_64_32_with16x32.cpp:266:9
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v086 v087 v088 v089, v138 v139 v140 v141, v062 v063 v064 v065	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v158 v159 v160 v161, v138 v139 v140 v141, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v158 v159 v160 v161, v146 v147 v148 v149, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v166 v167 v168 v169, v138 v139 v140 v141, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v166 v167 v168 v169, v146 v147 v148 v149, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v174 v175 v176 v177, v138 v139 v140 v141, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v174 v175 v176 v177, v146 v147 v148 v149, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v086 v087 v088 v089, v186 v187 v188 v189, v030 v031 v032 v033	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v086 v087 v088 v089, v194 v195 v196 v197, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v158 v159 v160 v161, v186 v187 v188 v189, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v158 v159 v160 v161, v194 v195 v196 v197, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v166 v167 v168 v169, v186 v187 v188 v189, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v166 v167 v168 v169, v194 v195 v196 v197, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v174 v175 v176 v177, v186 v187 v188 v189, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v174 v175 v176 v177, v194 v195 v196 v197, v002 v003 v004 v005
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v094 v095 v096 v097, v142 v143 v144 v145, v062 v063 v064 v065	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v086 v087 v088 v089, v146 v147 v148 v149, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v162 v163 v164 v165, v142 v143 v144 v145, v054 v055 v056 v057	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v162 v163 v164 v165, v150 v151 v152 v153, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v170 v171 v172 v173, v142 v143 v144 v145, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v170 v171 v172 v173, v150 v151 v152 v153, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v178 v179 v180 v181, v142 v143 v144 v145, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v178 v179 v180 v181, v150 v151 v152 v153, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v094 v095 v096 v097, v190 v191 v192 v193, v030 v031 v032 v033	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v094 v095 v096 v097, v198 v199 v200 v201, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v162 v163 v164 v165, v190 v191 v192 v193, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v162 v163 v164 v165, v198 v199 v200 v201, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v170 v171 v172 v173, v190 v191 v192 v193, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v170 v171 v172 v173, v198 v199 v200 v201, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v178 v179 v180 v181, v190 v191 v192 v193, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v178 v179 v180 v181, v198 v199 v200 v201, v002 v003 v004 v005
	v_mfma_f32_16x16x32_bf16 v182 v183 v184 v185, v094 v095 v096 v097, v150 v151 v152 v153, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	s_setprio 0	;.loc	2 269 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:269:9
	s_barrier	;.loc	2 270 9                         ; 256_256_64_32_with16x32.cpp:270:9
	s_nop 0	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	v_add3_u32 v058, s26, v135, v134
	ds_read_b128 v138 v139 v140 v141, v058 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v142 v143 v144 v145, v058 offset:0x400
	ds_read_b128 v146 v147 v148 v149, v058 offset:0x800
	ds_read_b128 v150 v151 v152 v153, v058 offset:0xc00
	v_add3_u32 v086, s25, v130, v134	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	ds_read_b128 v058 v059 v060 v061, v086 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v158 v159 v160 v161, v086 offset:0x400
	ds_read_b128 v162 v163 v164 v165, v086 offset:0x800
	ds_read_b128 v166 v167 v168 v169, v086 offset:0xc00
	ds_read_b128 v170 v171 v172 v173, v086 offset:0x1000
	ds_read_b128 v174 v175 v176 v177, v086 offset:0x1400
	ds_read_b128 v178 v179 v180 v181, v086 offset:0x1800
	ds_read_b128 v186 v187 v188 v189, v086 offset:0x1c00
	s_waitcnt vmcnt(2)	;.loc	2 279 9                         ; 256_256_64_32_with16x32.cpp:279:9
	s_barrier	;.loc	2 280 9                         ; 256_256_64_32_with16x32.cpp:280:9
	s_waitcnt lgkmcnt(0)	;.loc	2 282 9                         ; 256_256_64_32_with16x32.cpp:282:9
	s_setprio 1	;.loc	2 283 9                         ; 256_256_64_32_with16x32.cpp:283:9
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v058 v059 v060 v061, v138 v139 v140 v141, v126 v127 v128 v129	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v158 v159 v160 v161, v142 v143 v144 v145, v086 v087 v088 v089	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v058 v059 v060 v061, v146 v147 v148 v149, v122 v123 v124 v125	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v158 v159 v160 v161, v150 v151 v152 v153, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v162 v163 v164 v165, v138 v139 v140 v141, v118 v119 v120 v121	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v166 v167 v168 v169, v142 v143 v144 v145, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v162 v163 v164 v165, v146 v147 v148 v149, v114 v115 v116 v117	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v166 v167 v168 v169, v150 v151 v152 v153, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v170 v171 v172 v173, v138 v139 v140 v141, v110 v111 v112 v113	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v174 v175 v176 v177, v142 v143 v144 v145, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v170 v171 v172 v173, v146 v147 v148 v149, v106 v107 v108 v109	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v174 v175 v176 v177, v150 v151 v152 v153, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v178 v179 v180 v181, v138 v139 v140 v141, v102 v103 v104 v105	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v186 v187 v188 v189, v142 v143 v144 v145, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v178 v179 v180 v181, v146 v147 v148 v149, v098 v099 v100 v101	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v186 v187 v188 v189, v150 v151 v152 v153, v086 v087 v088 v089	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	s_setprio 0	;.loc	2 285 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:285:9
	s_barrier	;.loc	2 286 9                         ; 256_256_64_32_with16x32.cpp:286:9
	v_add3_u32 v098, s23, v135, v134	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	ds_read_b128 v190 v191 v192 v193, v098 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v194 v195 v196 v197, v098 offset:0x400
	ds_read_b128 v198 v199 v200 v201, v098 offset:0x800
	ds_read_b128 v206 v207 v208 v209, v098 offset:0xc00
	s_waitcnt vmcnt(0)	;.loc	2 290 9                         ; 256_256_64_32_with16x32.cpp:290:9
	s_barrier	;.loc	2 291 9                         ; 256_256_64_32_with16x32.cpp:291:9
	s_waitcnt lgkmcnt(0)	;.loc	2 293 9                         ; 256_256_64_32_with16x32.cpp:293:9
	s_setprio 1	;.loc	2 294 9                         ; 256_256_64_32_with16x32.cpp:294:9
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v058 v059 v060 v061, v190 v191 v192 v193, v202 v203 v204 v205	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v058 v059 v060 v061, v198 v199 v200 v201, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v158 v159 v160 v161, v194 v195 v196 v197, v098 v099 v100 v101	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v158 v159 v160 v161, v206 v207 v208 v209, v058 v059 v060 v061
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v162 v163 v164 v165, v190 v191 v192 v193, v154 v155 v156 v157	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v166 v167 v168 v169, v194 v195 v196 v197, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v162 v163 v164 v165, v198 v199 v200 v201, v082 v083 v084 v085	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v166 v167 v168 v169, v206 v207 v208 v209, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v170 v171 v172 v173, v190 v191 v192 v193, v078 v079 v080 v081	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v174 v175 v176 v177, v194 v195 v196 v197, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v170 v171 v172 v173, v198 v199 v200 v201, v074 v075 v076 v077	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v174 v175 v176 v177, v206 v207 v208 v209, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v178 v179 v180 v181, v190 v191 v192 v193, v070 v071 v072 v073	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v186 v187 v188 v189, v194 v195 v196 v197, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v178 v179 v180 v181, v198 v199 v200 v201, v066 v067 v068 v069	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v186 v187 v188 v189, v206 v207 v208 v209, v058 v059 v060 v061	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	s_setprio 0	;.loc	2 296 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:296:9
	s_barrier	;.loc	2 297 9                         ; 256_256_64_32_with16x32.cpp:297:9
	v_add3_u32 v066, s18, v130, v134	;.loc	26 62 51                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:62:51
	ds_read_b128 v132 v133 v134 v135, v066 offset:0	;.loc	26 78 37                        ; ../../..//include/ops/warp/memory/tile/shared_to_register.cuh:78:37
	ds_read_b128 v154 v155 v156 v157, v066 offset:0x400
	ds_read_b128 v158 v159 v160 v161, v066 offset:0x800
	ds_read_b128 v162 v163 v164 v165, v066 offset:0xc00
	ds_read_b128 v166 v167 v168 v169, v066 offset:0x1000
	ds_read_b128 v170 v171 v172 v173, v066 offset:0x1400
	ds_read_b128 v174 v175 v176 v177, v066 offset:0x1800
	ds_read_b128 v178 v179 v180 v181, v066 offset:0x1c00
	s_barrier	;.loc	2 301 9                         ; 256_256_64_32_with16x32.cpp:301:9
	s_waitcnt lgkmcnt(0)	;.loc	2 303 9                         ; 256_256_64_32_with16x32.cpp:303:9
	s_setprio 1	;.loc	2 304 9                         ; 256_256_64_32_with16x32.cpp:304:9
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v132 v133 v134 v135, v138 v139 v140 v141, v062 v063 v064 v065	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v154 v155 v156 v157, v142 v143 v144 v145, v062 v063 v064 v065	;.loc	16 35 22 is_stmt 0              ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v132 v133 v134 v135, v146 v147 v148 v149, v182 v183 v184 v185	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v158 v159 v160 v161, v138 v139 v140 v141, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v158 v159 v160 v161, v146 v147 v148 v149, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v166 v167 v168 v169, v138 v139 v140 v141, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v166 v167 v168 v169, v146 v147 v148 v149, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v174 v175 v176 v177, v138 v139 v140 v141, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v174 v175 v176 v177, v146 v147 v148 v149, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v132 v133 v134 v135, v190 v191 v192 v193, v030 v031 v032 v033	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v132 v133 v134 v135, v198 v199 v200 v201, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v158 v159 v160 v161, v190 v191 v192 v193, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v158 v159 v160 v161, v198 v199 v200 v201, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v166 v167 v168 v169, v190 v191 v192 v193, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v166 v167 v168 v169, v198 v199 v200 v201, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v174 v175 v176 v177, v190 v191 v192 v193, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v174 v175 v176 v177, v198 v199 v200 v201, v002 v003 v004 v005
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v154 v155 v156 v157, v150 v151 v152 v153, v062 v063 v064 v065	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v162 v163 v164 v165, v142 v143 v144 v145, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v162 v163 v164 v165, v150 v151 v152 v153, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v170 v171 v172 v173, v142 v143 v144 v145, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v170 v171 v172 v173, v150 v151 v152 v153, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v178 v179 v180 v181, v142 v143 v144 v145, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v178 v179 v180 v181, v150 v151 v152 v153, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v154 v155 v156 v157, v194 v195 v196 v197, v030 v031 v032 v033	;.loc	16 35 22                        ; ../../..//include/ops/warp/register/tile/mma.cuh:35:22
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v154 v155 v156 v157, v206 v207 v208 v209, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v162 v163 v164 v165, v194 v195 v196 v197, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v162 v163 v164 v165, v206 v207 v208 v209, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v170 v171 v172 v173, v194 v195 v196 v197, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v170 v171 v172 v173, v206 v207 v208 v209, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v178 v179 v180 v181, v194 v195 v196 v197, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v178 v179 v180 v181, v206 v207 v208 v209, v002 v003 v004 v005
	s_setprio 0	;.loc	2 307 9 is_stmt 1               ; 256_256_64_32_with16x32.cpp:307:9
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v000	;.loc	2 311 18                        ; 256_256_64_32_with16x32.cpp:311:18
	s_barrier	;.loc	2 308 9                         ; 256_256_64_32_with16x32.cpp:308:9
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 311 18                        ; 256_256_64_32_with16x32.cpp:311:18
.JUMP.LBB0_12:
	s_cbranch_execz .LBB0_12
	s_barrier	;.loc	2 312 9                         ; 256_256_64_32_with16x32.cpp:312:9
.LBB0_12:
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 9 is_stmt 0                 ; 256_256_64_32_with16x32.cpp:0:9
	v_lshl_or_b32 v168, s17, 2, v001	;.loc	2 316 29 is_stmt 1              ; 256_256_64_32_with16x32.cpp:316:29
	v_lshl_or_b32 v169, v131, 5, s19	;.loc	23 61 18                        ; ../../..//include/types/global/util.cuh:61:18
	s_waitcnt lgkmcnt(0)	;.loc	17 72 64                        ; ../../..//include/types/global/gl.cuh:72:64
	v_mul_lo_u32 v001, v168, s12
	v_lshl_add_u32 v130, v001, 6, v169	;.loc	17 72 72 is_stmt 0              ; ../../..//include/types/global/gl.cuh:72:72
	v_and_b32_e32 v001, 12, v136	;.loc	18 310 48 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:310:48
	v_and_b32_e32 v000, 15, v000	;.loc	18 311 34                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:311:34
	v_mad_u64_u32 v000 v001, s[0:1], v001, s12, v000 v001	;.loc	18 324 50                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshlrev_b64 v132 v133, 1, v000 v001
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v131, 31, v130	;.loc	17 72 16                        ; ../../..//include/types/global/gl.cuh:72:16
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_lshl_add_u64 v130 v131, v130 v131, 1, s[2:3]	;.loc	17 72 16                        ; ../../..//include/types/global/gl.cuh:72:16
	v_lshlrev_b64 v136 v137, 1, v000 v001	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 324 50                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_lshl_add_u64 v134 v135, v130 v131, 0, v132 v133	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v138 v139, v130 v131, 0, v136 v137	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v134 v135, v126, off	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v138 v139, v127, off	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshlrev_b64 v126 v127, 1, v000 v001	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	s_mul_i32 s0, s12, 13	;.loc	18 324 38 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:38
	v_lshlrev_b64 v142 v143, 1, v000 v001	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_add_u32_e32 v000, s0, v000	;.loc	18 324 50                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_lshl_add_u64 v140 v141, v130 v131, 0, v126 v127	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v144 v145, v130 v131, 0, v142 v143	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v140 v141, v128, off	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v144 v145, v129, off	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v134 v135, v122, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v138 v139, v123, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v140 v141, v124, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v144 v145, v125, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshlrev_b64 v122 v123, 1, v000 v001	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_lshlrev_b64 v128 v129, 1, v000 v001
	v_add_u32_e32 v000, s12, v000	;.loc	18 324 50 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_lshl_add_u64 v124 v125, v130 v131, 0, v122 v123	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v146 v147, v130 v131, 0, v128 v129	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v124 v125, v118, off	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v146 v147, v119, off	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshlrev_b64 v118 v119, 1, v000 v001	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_lshlrev_b64 v150 v151, 1, v000 v001
	v_add_u32_e32 v000, s0, v000	;.loc	18 324 50 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_lshl_add_u64 v148 v149, v130 v131, 0, v118 v119	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v152 v153, v130 v131, 0, v150 v151	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v148 v149, v120, off	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v152 v153, v121, off	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v124 v125, v114, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v146 v147, v115, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v148 v149, v116, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v152 v153, v117, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshlrev_b64 v114 v115, 1, v000 v001	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_lshlrev_b64 v120 v121, 1, v000 v001
	v_add_u32_e32 v000, s12, v000	;.loc	18 324 50 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_lshl_add_u64 v116 v117, v130 v131, 0, v114 v115	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v154 v155, v130 v131, 0, v120 v121	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v116 v117, v110, off	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v154 v155, v111, off	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshlrev_b64 v110 v111, 1, v000 v001	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_lshlrev_b64 v158 v159, 1, v000 v001
	v_add_u32_e32 v000, s0, v000	;.loc	18 324 50 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_lshl_add_u64 v156 v157, v130 v131, 0, v110 v111	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v160 v161, v130 v131, 0, v158 v159	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v156 v157, v112, off	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v160 v161, v113, off	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v116 v117, v106, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v154 v155, v107, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v156 v157, v108, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v160 v161, v109, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshlrev_b64 v106 v107, 1, v000 v001	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_lshlrev_b64 v112 v113, 1, v000 v001
	v_add_u32_e32 v000, s12, v000	;.loc	18 324 50 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:50
	v_lshl_add_u64 v108 v109, v130 v131, 0, v106 v107	;.loc	18 324 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v162 v163, v130 v131, 0, v112 v113	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v108 v109, v094, off	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v162 v163, v095, off	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshlrev_b64 v094 v095, 1, v000 v001	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_add_u32_e32 v000, s12, v000	;.loc	18 325 52                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:52
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	18 325 21 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	v_lshlrev_b64 v000 v001, 1, v000 v001
	v_lshl_add_u64 v164 v165, v130 v131, 0, v094 v095	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v166 v167, v130 v131, 0, v000 v001	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v164 v165, v096, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v166 v167, v097, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v108 v109, v086, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v162 v163, v087, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v164 v165, v088, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v166 v167, v089, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v086 v087, v130 v131, 0, s[0:1]	;.loc	17 72 16                        ; ../../..//include/types/global/gl.cuh:72:16
	v_lshl_add_u64 v088 v089, v086 v087, 0, v132 v133	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v134 v135, v102, off offset:256	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v138 v139, v103, off offset:256	;.loc	18 325 59 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v140 v141, v104, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v144 v145, v105, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v088 v089, v098, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v088 v089, v086 v087, 0, v136 v137	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v088 v089, v099, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v088 v089, v086 v087, 0, v126 v127	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v088 v089, v100, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v088 v089, v086 v087, 0, v142 v143	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v088 v089, v101, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v124 v125, v090, off offset:256	;.loc	18 324 57 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v146 v147, v091, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v148 v149, v092, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v152 v153, v093, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v088 v089, v086 v087, 0, v122 v123	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v088 v089, v082, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v088 v089, v086 v087, 0, v128 v129	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v088 v089, v083, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v082 v083, v086 v087, 0, v118 v119	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v082 v083, v084, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v082 v083, v086 v087, 0, v150 v151	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v082 v083, v085, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v116 v117, v078, off offset:256	;.loc	18 324 57 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v154 v155, v079, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v156 v157, v080, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v160 v161, v081, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v078 v079, v086 v087, 0, v114 v115	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v078 v079, v074, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v078 v079, v086 v087, 0, v120 v121	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v078 v079, v075, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v074 v075, v086 v087, 0, v110 v111	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v074 v075, v076, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v074 v075, v086 v087, 0, v158 v159	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v074 v075, v077, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v108 v109, v070, off offset:256	;.loc	18 324 57 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v162 v163, v071, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v164 v165, v072, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v166 v167, v073, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v070 v071, v086 v087, 0, v106 v107	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v070 v071, v058, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v070 v071, v086 v087, 0, v112 v113	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v070 v071, v059, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v058 v059, v086 v087, 0, v094 v095	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v058 v059, v060, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v058 v059, v086 v087, 0, v000 v001	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v058 v059, v061, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_or_b32_e32 v058, 2, v168	;.loc	2 322 39 is_stmt 1              ; 256_256_64_32_with16x32.cpp:322:39
	v_mul_lo_u32 v058, v058, s12	;.loc	17 72 64                        ; ../../..//include/types/global/gl.cuh:72:64
	v_lshl_add_u32 v058, v058, 6, v169	;.loc	17 72 72 is_stmt 0              ; ../../..//include/types/global/gl.cuh:72:72
	v_ashrrev_i32_e32 v059, 31, v058	;.loc	17 72 16                        ; ../../..//include/types/global/gl.cuh:72:16
	v_lshl_add_u64 v058 v059, v058 v059, 1, s[2:3]
	v_lshl_add_u64 v060 v061, v058 v059, 0, v132 v133	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v070 v071, v058 v059, 0, v136 v137	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v060 v061, v066, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v070 v071, v067, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v066 v067, v058 v059, 0, v126 v127	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v072 v073, v058 v059, 0, v142 v143	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v066 v067, v068, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v072 v073, v069, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v060 v061, v062, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v070 v071, v063, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v066 v067, v064, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v072 v073, v065, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v062 v063, v058 v059, 0, v122 v123	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v064 v065, v058 v059, 0, v128 v129	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v062 v063, v054, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v064 v065, v055, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v054 v055, v058 v059, 0, v118 v119	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v068 v069, v058 v059, 0, v150 v151	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v054 v055, v056, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v068 v069, v057, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v062 v063, v050, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v064 v065, v051, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v054 v055, v052, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v068 v069, v053, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v050 v051, v058 v059, 0, v114 v115	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v052 v053, v058 v059, 0, v120 v121	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v050 v051, v046, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v052 v053, v047, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v046 v047, v058 v059, 0, v110 v111	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v056 v057, v058 v059, 0, v158 v159	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v046 v047, v048, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v056 v057, v049, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v050 v051, v042, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v052 v053, v043, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v046 v047, v044, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v056 v057, v045, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v042 v043, v058 v059, 0, v106 v107	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v044 v045, v058 v059, 0, v112 v113	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v042 v043, v038, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v044 v045, v039, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v038 v039, v058 v059, 0, v094 v095	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v048 v049, v058 v059, 0, v000 v001	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v038 v039, v040, off	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v048 v049, v041, off	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v042 v043, v034, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v044 v045, v035, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v038 v039, v036, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v048 v049, v037, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v034 v035, v058 v059, 0, s[0:1]	;.loc	17 72 16                        ; ../../..//include/types/global/gl.cuh:72:16
	global_store_short_d16_hi v060 v061, v030, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v070 v071, v031, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v066 v067, v032, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v072 v073, v033, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v030 v031, v034 v035, 0, v132 v133	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v030 v031, v026, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v030 v031, v034 v035, 0, v136 v137	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v030 v031, v027, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v026 v027, v034 v035, 0, v126 v127	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v026 v027, v028, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v026 v027, v034 v035, 0, v142 v143	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v026 v027, v029, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v062 v063, v022, off offset:256	;.loc	18 324 57 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v064 v065, v023, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v054 v055, v024, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v068 v069, v025, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v022 v023, v034 v035, 0, v122 v123	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v022 v023, v018, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v022 v023, v034 v035, 0, v128 v129	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v022 v023, v019, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v018 v019, v034 v035, 0, v118 v119	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v018 v019, v020, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v018 v019, v034 v035, 0, v150 v151	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v018 v019, v021, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v050 v051, v014, off offset:256	;.loc	18 324 57 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v052 v053, v015, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v046 v047, v016, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v056 v057, v017, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v014 v015, v034 v035, 0, v114 v115	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v014 v015, v010, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v014 v015, v034 v035, 0, v120 v121	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v014 v015, v011, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v010 v011, v034 v035, 0, v110 v111	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v010 v011, v012, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v010 v011, v034 v035, 0, v158 v159	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v010 v011, v013, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v042 v043, v006, off offset:256	;.loc	18 324 57 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v044 v045, v007, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	global_store_short_d16_hi v038 v039, v008, off offset:256	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v048 v049, v009, off offset:256	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v006 v007, v034 v035, 0, v106 v107	;.loc	18 324 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	global_store_short_d16_hi v006 v007, v002, off offset:32	;.loc	18 324 57 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	v_lshl_add_u64 v006 v007, v034 v035, 0, v112 v113	;.loc	18 325 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v006 v007, v003, off offset:32	;.loc	18 325 59 is_stmt 0             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	v_lshl_add_u64 v002 v003, v034 v035, 0, v094 v095	;.loc	18 324 21 is_stmt 1             ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:21
	v_lshl_add_u64 v000 v001, v034 v035, 0, v000 v001	;.loc	18 325 21                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:21
	global_store_short_d16_hi v002 v003, v004, off offset:32	;.loc	18 324 57                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:324:57
	global_store_short_d16_hi v000 v001, v005, off offset:32	;.loc	18 325 59                       ; ../../..//include/ops/warp/memory/tile/global_to_register.cuh:325:59
	s_endpgm	;.loc	2 327 1                         ; 256_256_64_32_with16x32.cpp:327:1
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
...
	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
