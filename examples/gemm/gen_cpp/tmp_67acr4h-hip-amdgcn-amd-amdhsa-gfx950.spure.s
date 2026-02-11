	.file	0 "/A/tilelang/examples/gemm" "tmp_67acr4h.cpp" md5 0x1f530572506a2be7a62a715bd8bfcaac
	.file	1 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "host_defines.h"
	.file	2 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_bfloat16.h"
	.file	3 "/A/tilelang" "3rdparty/../src/tl_templates/hip/common.h"
	.file	4 "/tmp" "tmp_67acr4h.cpp"
	.file	5 "/usr/include/x86_64-linux-gnu/bits" "types.h"
	.file	6 "/usr/include/x86_64-linux-gnu/bits" "stdint-uintn.h"
	.file	7 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/rocwmma/internal/layout/.." "types.hpp"
	.file	8 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/rocwmma/internal" "cross_lane_ops.hpp"
	.file	9 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/rocwmma/internal/layout" "layout.hpp"
	.file	10 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_fp8.h"
	.file	11 "/A/tilelang" "3rdparty/../src/tl_templates/hip/hip_fp8.h"
	.file	12 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_vector_types.h"
	.file	13 "/opt/rocm-7.2.0/lib/llvm/lib/clang/20/include" "__stddef_size_t.h"
	.file	14 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_bf16.h"
	.file	15 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_fp16.h"
	.file	16 "/usr/include" "stdint.h"
	.file	17 "/A/tilelang" "3rdparty/../src/tl_templates/hip/copy.h"
	.file	18 "/usr/include/x86_64-linux-gnu/bits" "stdint-intn.h"
	.file	19 "/A/tilelang" "3rdparty/composable_kernel/include/ck_tile/core/numeric/integer.hpp"
	.file	20 "/A/tilelang" "3rdparty/composable_kernel/include/ck_tile/core/numeric/vector_type.hpp"
	.file	21 "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip" "hip_runtime_api.h"
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	gemm_kernel             ; -- Begin function gemm_kernel
	.globl	gemm_kernel
	.p2align	8
	.type	gemm_kernel,@function
gemm_kernel:                            ; @gemm_kernel
	.file	22 "/A/tilelang" "3rdparty/../src/tl_templates/hip/threadblock_swizzle.h"
	s_waitcnt lgkmcnt(0)	;.loc	22 91 52 prologue_end           ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:91:52
	s_mul_i32 s0, s10, s3
	s_add_i32 s0, s0, s2	;.loc	22 91 39 is_stmt 0              ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:91:39
	s_lshl_b32 s12, s0, 4
	s_and_b32 s3, s0, 0xffffff80
	s_and_b32 s12, s12, 0x70
	s_mul_i32 s1, s11, s10	;.loc	22 88 44 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:88:44
	s_bfe_u32 s11, s0, 0x40003	;.loc	22 0 44 is_stmt 0               ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:0:44
	s_or_b32 s3, s12, s3
	s_and_b32 s2, s1, 0xffffff80	;.loc	22 22 40 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:22:40
	s_or_b32 s3, s3, s11	;.loc	22 0 40 is_stmt 0               ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:0:40
	s_cmp_lt_u32 s0, s2	;.loc	22 25 20 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:25:20
	s_cselect_b32 s12, s3, s0
	s_lshl_b32 s13, s10, 2	;.loc	22 96 47                        ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:96:47
	v_cvt_f32_u32_e32 v001, s13	;.loc	22 98 44                        ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:98:44
	s_sub_i32 s0, 0, s13
	v_rcp_iflag_f32_e32 v001, v001
	s_nop 0
	v_mul_f32_e32 v001, 0x4f7ffffe, v001
	v_cvt_u32_f32_e32 v001, v001
	s_nop 0
	v_readfirstlane_b32 s2, v001
	s_mul_i32 s0, s0, s2
	s_mul_hi_u32 s0, s2, s0
	s_add_i32 s2, s2, s0
	s_mul_hi_u32 s0, s12, s2
	s_mul_i32 s2, s0, s13
	s_sub_i32 s2, s12, s2
	s_add_i32 s3, s0, 1
	s_sub_i32 s11, s2, s13
	s_cmp_ge_u32 s2, s13
	s_cselect_b32 s0, s3, s0
	s_cselect_b32 s2, s11, s2
	s_add_i32 s3, s0, 1
	s_cmp_ge_u32 s2, s13
	s_cselect_b32 s16, s3, s0
	s_abs_i32 s3, s13	;.loc	22 87 57                        ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:87:57
	v_cvt_f32_u32_e32 v001, s3
	s_sub_i32 s15, 0, s3
	s_add_i32 s14, s13, s1	;.loc	22 87 48 is_stmt 0              ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:87:48
	s_add_i32 s14, s14, -1	;.loc	22 87 52                        ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:87:52
	v_rcp_iflag_f32_e32 v001, v001	;.loc	22 87 57                        ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:87:57
	s_xor_b32 s17, s14, s13
	s_abs_i32 s14, s14
	s_ashr_i32 s17, s17, 31
	v_mul_f32_e32 v001, 0x4f7ffffe, v001
	v_cvt_u32_f32_e32 v001, v001
	s_mov_b32 s0, 4
	s_mov_b32 s11, 0
	s_mov_b32 s2, -1
	v_readfirstlane_b32 s18, v001
	s_mul_i32 s15, s15, s18
	s_mul_hi_u32 s15, s18, s15
	s_add_i32 s18, s18, s15
	s_mul_hi_u32 s15, s14, s18
	s_mul_i32 s18, s15, s3
	s_sub_i32 s14, s14, s18
	s_add_i32 s19, s15, 1
	s_sub_i32 s18, s14, s3
	s_cmp_ge_u32 s14, s3
	s_cselect_b32 s15, s19, s15
	s_cselect_b32 s14, s18, s14
	s_add_i32 s18, s15, 1
	s_cmp_ge_u32 s14, s3
	s_cselect_b32 s3, s18, s15
	s_xor_b32 s3, s3, s17
	s_sub_i32 s3, s3, s17
	s_add_i32 s14, s16, 1	;.loc	22 101 17 is_stmt 1             ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:101:17
	s_cmp_lt_u32 s14, s3	;.loc	22 101 21 is_stmt 0             ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:101:21
	s_mul_i32 s3, s16, s13
.JUMP.LBB0_2:
	s_cbranch_scc1 .LBB0_2	;.loc	22 101 7                        ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:101:7
	v_cvt_f32_u32_e32 v001, s10	;.loc	22 103 50 is_stmt 1             ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:103:50
	s_sub_i32 s0, s1, s3
	s_sub_i32 s1, 0, s10
	v_rcp_iflag_f32_e32 v001, v001
	s_nop 0
	v_mul_f32_e32 v001, 0x4f7ffffe, v001
	v_cvt_u32_f32_e32 v001, v001
	s_nop 0
	v_readfirstlane_b32 s14, v001
	s_mul_i32 s1, s1, s14
	s_mul_hi_u32 s1, s14, s1
	s_add_i32 s14, s14, s1
	s_mul_hi_u32 s1, s0, s14
	s_mul_i32 s14, s1, s10
	s_sub_i32 s0, s0, s14
	s_add_i32 s15, s1, 1
	s_sub_i32 s14, s0, s10
	s_cmp_ge_u32 s0, s10
	s_cselect_b32 s1, s15, s1
	s_cselect_b32 s0, s14, s0
	s_add_i32 s14, s1, 1
	s_cmp_ge_u32 s0, s10
	s_cselect_b32 s0, s14, s1
.LBB0_2:                                ; %_ZN2tl21rasterization2DRowXcdILi4ELi8EEE4dim3v.exit
	v_cvt_f32_u32_e32 v001, s0	;.loc	22 0 0 is_stmt 0                ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:0:0
	s_sub_i32 s1, s12, s3	;.loc	22 97 47 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:97:47
	s_sub_i32 s3, 0, s0	;.loc	22 0 0 is_stmt 0                ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:0:0
	v_lshlrev_b32_e32 v003, 10, v000
	v_rcp_iflag_f32_e32 v001, v001
	v_and_b32_e32 v006, 0x7e000, v003
	v_lshlrev_b32_e32 v130, 4, v000	;.loc	4 24 28 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:24:28
	s_mov_b32 m0, s11	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	v_mul_f32_e32 v001, 0x4f7ffffe, v001	;.loc	22 0 0 is_stmt 0                ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:0:0
	v_cvt_u32_f32_e32 v002, v001
	v_lshlrev_b32_e32 v001, 3, v000
	v_readfirstlane_b32 s18, v130	;.loc	17 114 23 is_stmt 1             ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	s_mov_b32 m0, s18	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_readfirstlane_b32 s14, v002	;.loc	22 0 0 is_stmt 0                ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:0:0
	s_mul_i32 s3, s3, s14
	s_mul_hi_u32 s3, s14, s3
	s_add_i32 s14, s14, s3
	s_mul_hi_u32 s3, s1, s14
	s_mul_i32 s14, s3, s0
	s_sub_i32 s14, s1, s14
	s_add_i32 s15, s3, 1
	s_sub_i32 s17, s14, s0
	s_cmp_ge_u32 s14, s0
	s_cselect_b32 s3, s15, s3
	s_cselect_b32 s14, s17, s14
	s_add_i32 s15, s3, 1
	v_and_b32_e32 v002, 32, v001
	s_cmp_ge_u32 s14, s0
	v_add_u32_e32 v007, v002, v000
	v_and_b32_e32 v002, 16, v001
	s_cselect_b32 s15, s15, s3
	v_add_u32_e32 v002, v002, v000
	s_mul_i32 s17, s15, s0	;.loc	22 107 45 is_stmt 1             ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:107:45
	v_and_b32_e32 v135, 16, v002
	v_mul_u32_u24_e32 v002, 9, v000
	s_sub_i32 s0, s1, s17
	v_and_b32_e32 v008, 8, v002
	s_lshl_b32 s14, s16, 23
	s_lshl_b32 s0, s0, 21
	v_and_b32_e32 v134, 32, v007
	v_or_b32_e32 v002, v008, v006
	s_add_i32 s14, s14, s0
	v_or3_b32 v009, v002, v134, v135
	v_or_b32_e32 v002, s14, v009
	v_ashrrev_i32_e32 v003, 31, v002	;.loc	4 24 86                         ; /tmp/tmp_67acr4h.cpp:24:86
	s_mov_b64 s[0:1], src_shared_base	;.loc	4 24 28 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:24:28
	v_lshl_add_u64 v004 v005, v002 v003, 1, s[4:5]	;.loc	4 24 86                         ; /tmp/tmp_67acr4h.cpp:24:86
	s_mov_b64 s[18:19], 0x100000
	v_readfirstlane_b32 s0, v004	;.loc	17 29 9 is_stmt 1               ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_mov_b32_e32 v131, s1	;.loc	4 24 28                         ; /tmp/tmp_67acr4h.cpp:24:28
	v_readfirstlane_b32 s1, v005	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	s_mov_b32 s3, 0x20000
	v_subrev_u32_e32 v003, s0, v004	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_lshl_add_u64 v004 v005, v004 v005, 0, s[18:19]	;.loc	4 24 86                         ; /tmp/tmp_67acr4h.cpp:24:86
	v_or_b32_e32 v132, 0x10000, v130	;.loc	4 29 28                         ; /tmp/tmp_67acr4h.cpp:29:28
	v_lshlrev_b32_e32 v139, 6, v000
	buffer_load_dwordx4 v003, s[0:3], 0 offen lds	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	v_or_b32_e32 v003, 0x2000, v130	;.loc	4 24 28                         ; /tmp/tmp_67acr4h.cpp:24:28
	v_readfirstlane_b32 s0, v004	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s21, v003	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_readfirstlane_b32 s1, v005	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v003, s0, v004	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_or_b32_e32 v004, 0x100000, v002	;.loc	4 24 349                        ; /tmp/tmp_67acr4h.cpp:24:349
	v_ashrrev_i32_e32 v005, 31, v004	;.loc	4 24 86 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:24:86
	s_mov_b32 m0, s21	;.loc	17 131 5 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_lshl_add_u64 v004 v005, v004 v005, 1, s[4:5]	;.loc	4 24 86                         ; /tmp/tmp_67acr4h.cpp:24:86
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v003, s[0:3], 0 offen lds
	v_or_b32_e32 v003, 0x4000, v130	;.loc	4 24 28                         ; /tmp/tmp_67acr4h.cpp:24:28
	v_readfirstlane_b32 s0, v004	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s21, v003	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_readfirstlane_b32 s1, v005	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v003, s0, v004	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	s_mov_b32 m0, s21	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_or_b32_e32 v002, 0x180000, v002	;.loc	4 24 349                        ; /tmp/tmp_67acr4h.cpp:24:349
	v_or_b32_e32 v004, 0x6000, v130	;.loc	4 24 28 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:24:28
	s_movk_i32 s20, 0x2000
	s_nop 0	;.loc	17 132 5 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v003, s[0:3], 0 offen lds
	v_ashrrev_i32_e32 v003, 31, v002	;.loc	4 24 86                         ; /tmp/tmp_67acr4h.cpp:24:86
	v_lshl_add_u64 v002 v003, v002 v003, 1, s[4:5]
	v_readfirstlane_b32 s21, v004	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_readfirstlane_b32 s0, v002	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s1, v003	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	s_mov_b32 m0, s21	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_readfirstlane_b32 s21, v132	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_subrev_u32_e32 v002, s0, v002	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_mov_b32_e32 v094, 0	;.loc	4 33 5                          ; /tmp/tmp_67acr4h.cpp:33:5
	v_mov_b32_e32 v133, v131	;.loc	4 29 28                         ; /tmp/tmp_67acr4h.cpp:29:28
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v002, s[0:3], 0 offen lds
	s_not_b32 s1, s15	;.loc	22 104 32                       ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:104:32
	s_and_b32 s0, s16, 1	;.loc	22 104 43 is_stmt 0             ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:104:43
	s_add_i32 s1, s10, s1	;.loc	22 104 32                       ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:104:32
	s_cmp_eq_u32 s0, 0
	s_cselect_b32 s15, s15, s1
	s_lshl_b32 s10, s15, 21	;.loc	22 0 32                         ; 3rdparty/../src/tl_templates/hip/threadblock_swizzle.h:0:32
	v_or_b32_e32 v002, s10, v009
	v_ashrrev_i32_e32 v003, 31, v002	;.loc	4 29 86 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:29:86
	v_lshl_add_u64 v004 v005, v002 v003, 1, s[6:7]
	s_barrier	;.loc	4 26 3                          ; /tmp/tmp_67acr4h.cpp:26:3
	v_readfirstlane_b32 s0, v004	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s1, v005	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	s_mov_b32 m0, s21	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_and_b32_e32 v138, 8, v001
	v_subrev_u32_e32 v003, s0, v004	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_lshl_add_u64 v004 v005, v004 v005, 0, s[18:19]	;.loc	4 29 86                         ; /tmp/tmp_67acr4h.cpp:29:86
	v_bitop3_b32 v136, v001, 8, v001 bitop3:0xc	;.loc	4 0 86 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:86
	s_nop 0	;.loc	17 132 5 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v003, s[0:3], 0 offen lds
	v_or_b32_e32 v003, 0x12000, v130	;.loc	4 29 28                         ; /tmp/tmp_67acr4h.cpp:29:28
	v_readfirstlane_b32 s0, v004	;.loc	17 29 9 is_stmt 0               ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s18, v003	;.loc	17 114 23 is_stmt 1             ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_readfirstlane_b32 s1, v005	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v003, s0, v004	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_or_b32_e32 v004, 0x100000, v002	;.loc	4 29 349                        ; /tmp/tmp_67acr4h.cpp:29:349
	v_ashrrev_i32_e32 v005, 31, v004	;.loc	4 29 86 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:29:86
	s_mov_b32 m0, s18	;.loc	17 131 5 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_lshl_add_u64 v004 v005, v004 v005, 1, s[6:7]	;.loc	4 29 86                         ; /tmp/tmp_67acr4h.cpp:29:86
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v003, s[0:3], 0 offen lds
	v_or_b32_e32 v003, 0x14000, v130	;.loc	4 29 28                         ; /tmp/tmp_67acr4h.cpp:29:28
	v_readfirstlane_b32 s0, v004	;.loc	17 29 9 is_stmt 0               ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s18, v003	;.loc	17 114 23 is_stmt 1             ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_readfirstlane_b32 s1, v005	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v003, s0, v004	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	s_mov_b32 m0, s18	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_or_b32_e32 v002, 0x180000, v002	;.loc	4 29 349                        ; /tmp/tmp_67acr4h.cpp:29:349
	v_or_b32_e32 v004, 0x16000, v130	;.loc	4 29 28 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:29:28
	v_mov_b32_e32 v143, 0x4000	;.loc	4 0 28                          ; /tmp/tmp_67acr4h.cpp:0:28
	s_nop 0	;.loc	17 132 5 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v003, s[0:3], 0 offen lds
	v_ashrrev_i32_e32 v003, 31, v002	;.loc	4 29 86                         ; /tmp/tmp_67acr4h.cpp:29:86
	v_lshl_add_u64 v002 v003, v002 v003, 1, s[6:7]
	v_readfirstlane_b32 s18, v004	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_readfirstlane_b32 s0, v002	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s1, v003	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	s_mov_b32 m0, s18	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_lshlrev_b32_e32 v003, 5, v000
	v_subrev_u32_e32 v002, s0, v002	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_mov_b32_e32 v144, 0x6000	;.loc	17 0 30 is_stmt 0               ; 3rdparty/../src/tl_templates/hip/copy.h:0:30
	v_mov_b32_e32 v145, 0x10000
	s_nop 0	;.loc	17 132 5 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v002, s[0:3], 0 offen lds
	v_and_b32_e32 v002, 0x3c0, v139	;.loc	17 0 5 is_stmt 0                ; 3rdparty/../src/tl_templates/hip/copy.h:0:5
	v_and_or_b32 v003, v003, s20, v002	;.loc	4 32 3 is_stmt 1                ; /tmp/tmp_67acr4h.cpp:32:3
	s_movk_i32 s0, 0x3020
	v_or3_b32 v137, v003, v134, v135
	v_bitop3_b32 v003, v007, s0, v139 bitop3:0xc8
	s_lshl_b32 s0, s13, 21
	s_sub_i32 s0, 0x800000, s0
	s_mul_i32 s0, s16, s0
	s_lshl_b32 s1, s12, 21
	v_or3_b32 v140, v003, v002, v135
	v_or3_b32 v002, s10, v006, v134
	s_add_i32 s0, s0, s1
	v_or3_b32 v141, v002, v135, v008
	v_or_b32_e32 v002, s0, v006
	v_or3_b32 v002, v002, v134, v135
	v_add_u32_e32 v002, v002, v008
	s_lshl_b32 s0, s17, 21
	v_subrev_u32_e32 v142, s0, v002
	s_mov_b64 s[12:13], 0x80	;.loc	4 0 3 is_stmt 0                 ; /tmp/tmp_67acr4h.cpp:0:3
	s_mov_b32 s16, 0
	s_mov_b32 s17, 0
	v_mov_b32_e32 v095, v094
	v_mov_b32_e32 v096, v094
	v_mov_b32_e32 v097, v094
	v_mov_b32_e32 v002, v094
	v_mov_b32_e32 v003, v094
	v_mov_b32_e32 v004, v094
	v_mov_b32_e32 v005, v094
	v_mov_b32_e32 v006, v094
	v_mov_b32_e32 v007, v094
	v_mov_b32_e32 v008, v094
	v_mov_b32_e32 v009, v094
	v_mov_b32_e32 v010, v094
	v_mov_b32_e32 v011, v094
	v_mov_b32_e32 v012, v094
	v_mov_b32_e32 v013, v094
	v_mov_b32_e32 v014, v094
	v_mov_b32_e32 v015, v094
	v_mov_b32_e32 v016, v094
	v_mov_b32_e32 v017, v094
	v_mov_b32_e32 v018, v094
	v_mov_b32_e32 v019, v094
	v_mov_b32_e32 v020, v094
	v_mov_b32_e32 v021, v094
	v_mov_b32_e32 v022, v094
	v_mov_b32_e32 v023, v094
	v_mov_b32_e32 v024, v094
	v_mov_b32_e32 v025, v094
	v_mov_b32_e32 v026, v094
	v_mov_b32_e32 v027, v094
	v_mov_b32_e32 v028, v094
	v_mov_b32_e32 v029, v094
	v_mov_b32_e32 v030, v094
	v_mov_b32_e32 v031, v094
	v_mov_b32_e32 v032, v094
	v_mov_b32_e32 v033, v094
	v_mov_b32_e32 v034, v094
	v_mov_b32_e32 v035, v094
	v_mov_b32_e32 v036, v094
	v_mov_b32_e32 v037, v094
	v_mov_b32_e32 v038, v094
	v_mov_b32_e32 v039, v094
	v_mov_b32_e32 v040, v094
	v_mov_b32_e32 v041, v094
	v_mov_b32_e32 v042, v094
	v_mov_b32_e32 v043, v094
	v_mov_b32_e32 v044, v094
	v_mov_b32_e32 v045, v094
	v_mov_b32_e32 v046, v094
	v_mov_b32_e32 v047, v094
	v_mov_b32_e32 v048, v094
	v_mov_b32_e32 v049, v094
	v_mov_b32_e32 v050, v094
	v_mov_b32_e32 v051, v094
	v_mov_b32_e32 v052, v094
	v_mov_b32_e32 v053, v094
	v_mov_b32_e32 v054, v094
	v_mov_b32_e32 v055, v094
	v_mov_b32_e32 v056, v094
	v_mov_b32_e32 v057, v094
	v_mov_b32_e32 v058, v094
	v_mov_b32_e32 v059, v094
	v_mov_b32_e32 v060, v094
	v_mov_b32_e32 v061, v094
	v_mov_b32_e32 v062, v094
	v_mov_b32_e32 v063, v094
	v_mov_b32_e32 v064, v094
	v_mov_b32_e32 v065, v094
	v_mov_b32_e32 v066, v094
	v_mov_b32_e32 v067, v094
	v_mov_b32_e32 v068, v094
	v_mov_b32_e32 v069, v094
	v_mov_b32_e32 v070, v094
	v_mov_b32_e32 v071, v094
	v_mov_b32_e32 v072, v094
	v_mov_b32_e32 v073, v094
	v_mov_b32_e32 v074, v094
	v_mov_b32_e32 v075, v094
	v_mov_b32_e32 v076, v094
	v_mov_b32_e32 v077, v094
	v_mov_b32_e32 v078, v094
	v_mov_b32_e32 v079, v094
	v_mov_b32_e32 v080, v094
	v_mov_b32_e32 v081, v094
	v_mov_b32_e32 v082, v094
	v_mov_b32_e32 v083, v094
	v_mov_b32_e32 v084, v094
	v_mov_b32_e32 v085, v094
	v_mov_b32_e32 v086, v094
	v_mov_b32_e32 v087, v094
	v_mov_b32_e32 v088, v094
	v_mov_b32_e32 v089, v094
	v_mov_b32_e32 v090, v094
	v_mov_b32_e32 v091, v094
	v_mov_b32_e32 v092, v094
	v_mov_b32_e32 v093, v094
	v_mov_b32_e32 v098, v094
	v_mov_b32_e32 v099, v094
	v_mov_b32_e32 v100, v094
	v_mov_b32_e32 v101, v094
	v_mov_b32_e32 v102, v094
	v_mov_b32_e32 v103, v094
	v_mov_b32_e32 v104, v094
	v_mov_b32_e32 v105, v094
	v_mov_b32_e32 v106, v094
	v_mov_b32_e32 v107, v094
	v_mov_b32_e32 v108, v094
	v_mov_b32_e32 v109, v094
	v_mov_b32_e32 v110, v094
	v_mov_b32_e32 v111, v094
	v_mov_b32_e32 v112, v094
	v_mov_b32_e32 v113, v094
	v_mov_b32_e32 v114, v094
	v_mov_b32_e32 v115, v094
	v_mov_b32_e32 v116, v094
	v_mov_b32_e32 v117, v094
	v_mov_b32_e32 v118, v094
	v_mov_b32_e32 v119, v094
	v_mov_b32_e32 v120, v094
	v_mov_b32_e32 v121, v094
	v_mov_b32_e32 v122, v094
	v_mov_b32_e32 v123, v094
	v_mov_b32_e32 v124, v094
	v_mov_b32_e32 v125, v094
	v_mov_b32_e32 v126, v094
	v_mov_b32_e32 v127, v094
	v_mov_b32_e32 v128, v094
	v_mov_b32_e32 v129, v094
	s_mov_b32 m0, 0
.LBB0_3:                                ; %.preheader166
	s_and_b32 s0, s16, 0x4000
	v_add_u32_e32 v146, s17, v142	;.loc	4 36 81 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:36:81
	v_bitop3_b32 v147, s16, v143, v001 bitop3:0x26
	v_lshlrev_b32_e32 v157, 1, v147	;.loc	4 36 30 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:36:30
	v_ashrrev_i32_e32 v147, 31, v146	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	s_xor_b32 s10, s0, 0x5000	;.loc	4 36 65                         ; /tmp/tmp_67acr4h.cpp:36:65
	v_add_u32_e32 v150, 0x80000, v146	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_bitop3_b32 v153, s0, v144, v001 bitop3:0x36	;.loc	4 36 81                         ; /tmp/tmp_67acr4h.cpp:36:81
	v_add_u32_e32 v152, 0x100000, v146	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	s_xor_b32 s18, s0, 0x7000	;.loc	4 36 65                         ; /tmp/tmp_67acr4h.cpp:36:65
	v_add_u32_e32 v154, 0x180000, v146	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_lshl_add_u64 v146 v147, v146 v147, 1, s[4:5]
	s_lshl_b32 s10, s10, 1	;.loc	4 36 30                         ; /tmp/tmp_67acr4h.cpp:36:30
	s_mov_b32 s1, s11
	v_or_b32_e32 v194, s0, v140
	v_or_b32_e32 v195, s0, v137
	v_ashrrev_i32_e32 v151, 31, v150	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_lshlrev_b32_e32 v168, 1, v153	;.loc	4 36 30                         ; /tmp/tmp_67acr4h.cpp:36:30
	s_lshl_b32 s0, s18, 1
	v_lshl_add_u64 v146 v147, v146 v147, 0, s[12:13]	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_lshl_add_u64 v162 v163, v130 v131, 0, s[10:11]	;.loc	4 36 30                         ; /tmp/tmp_67acr4h.cpp:36:30
	v_lshl_add_u64 v150 v151, v150 v151, 1, s[4:5]	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_readfirstlane_b32 s18, v168	;.loc	17 114 23 is_stmt 1             ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_lshl_add_u64 v164 v165, v130 v131, 0, s[0:1]	;.loc	4 36 30                         ; /tmp/tmp_67acr4h.cpp:36:30
	v_or_b32_e32 v163, 0x10000, v168	;.loc	4 41 30                         ; /tmp/tmp_67acr4h.cpp:41:30
	v_lshl_add_u64 v168 v169, v132 v133, 0, s[0:1]
	v_readfirstlane_b32 s0, v146	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	s_barrier	;.loc	4 33 5                          ; /tmp/tmp_67acr4h.cpp:33:5
	v_readfirstlane_b32 s19, v157	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_ashrrev_i32_e32 v153, 31, v152	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	s_mov_b32 m0, s19	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_readfirstlane_b32 s1, v147	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_lshl_add_u64 v150 v151, v150 v151, 0, s[12:13]	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_subrev_u32_e32 v146, s0, v146	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_or_b32_e32 v166, 0x10000, v157	;.loc	4 41 30                         ; /tmp/tmp_67acr4h.cpp:41:30
	v_lshl_add_u64 v152 v153, v152 v153, 1, s[4:5]	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_ashrrev_i32_e32 v155, 31, v154
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	v_readfirstlane_b32 s0, v150	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s19, v166	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_lshl_add_u64 v166 v167, v132 v133, 0, s[10:11]	;.loc	4 41 30                         ; /tmp/tmp_67acr4h.cpp:41:30
	v_readfirstlane_b32 s10, v162	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_lshl_add_u64 v152 v153, v152 v153, 0, s[12:13]	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_readfirstlane_b32 s1, v151	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v146, s0, v150	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	s_mov_b32 m0, s10	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_add_u32_e32 v148, s17, v141	;.loc	4 41 30                         ; /tmp/tmp_67acr4h.cpp:41:30
	v_lshl_add_u64 v154 v155, v154 v155, 1, s[4:5]	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_ashrrev_i32_e32 v149, 31, v148	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_readfirstlane_b32 s0, v152	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_lshl_add_u64 v154 v155, v154 v155, 0, s[12:13]	;.loc	4 36 116                        ; /tmp/tmp_67acr4h.cpp:36:116
	v_readfirstlane_b32 s1, v153	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v146, s0, v152	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	s_mov_b32 m0, s18	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_add_u32_e32 v156, 0x80000, v148	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	v_add_u32_e32 v158, 0x100000, v148
	v_add_u32_e32 v160, 0x180000, v148
	v_lshl_add_u64 v148 v149, v148 v149, 1, s[6:7]
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	v_readfirstlane_b32 s0, v154	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_ashrrev_i32_e32 v157, 31, v156	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	v_lshl_add_u64 v148 v149, v148 v149, 0, s[12:13]
	v_readfirstlane_b32 s20, v164	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_readfirstlane_b32 s1, v155	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v146, s0, v154	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	s_mov_b32 m0, s20	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_lshl_add_u64 v156 v157, v156 v157, 1, s[6:7]	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	v_ashrrev_i32_e32 v159, 31, v158
	v_lshl_add_u64 v156 v157, v156 v157, 0, s[12:13]
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_readfirstlane_b32 s0, v148	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	s_barrier	;.loc	4 38 5                          ; /tmp/tmp_67acr4h.cpp:38:5
	v_readfirstlane_b32 s1, v149	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	s_mov_b32 m0, s19	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_subrev_u32_e32 v146, s0, v148	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_lshl_add_u64 v158 v159, v158 v159, 1, s[6:7]	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	v_ashrrev_i32_e32 v161, 31, v160
	v_readfirstlane_b32 s21, v166	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_readfirstlane_b32 s0, v156	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_lshl_add_u64 v158 v159, v158 v159, 0, s[12:13]	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	v_readfirstlane_b32 s1, v157	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	v_subrev_u32_e32 v146, s0, v156	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	s_mov_b32 m0, s21	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_lshl_add_u64 v160 v161, v160 v161, 1, s[6:7]	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	v_readfirstlane_b32 s22, v163	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_lshl_add_u64 v160 v161, v160 v161, 0, s[12:13]	;.loc	4 41 116                        ; /tmp/tmp_67acr4h.cpp:41:116
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_readfirstlane_b32 s0, v158	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s1, v159	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	s_mov_b32 m0, s22	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_readfirstlane_b32 s23, v168	;.loc	17 114 23                       ; 3rdparty/../src/tl_templates/hip/copy.h:114:23
	v_subrev_u32_e32 v146, s0, v158	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_or_b32_e32 v170, v138, v194	;.loc	4 48 375                        ; /tmp/tmp_67acr4h.cpp:48:375
	v_or_b32_e32 v171, v136, v194
	s_nop 0	;.loc	17 132 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_readfirstlane_b32 s0, v160	;.loc	17 29 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:29:9
	v_readfirstlane_b32 s1, v161	;.loc	17 30 9                         ; 3rdparty/../src/tl_templates/hip/copy.h:30:9
	s_mov_b32 m0, s23	;.loc	17 131 5                        ; 3rdparty/../src/tl_templates/hip/copy.h:131:5
	v_or_b32_e32 v172, v138, v195	;.loc	4 53 365                        ; /tmp/tmp_67acr4h.cpp:53:365
	v_subrev_u32_e32 v146, s0, v160	;.loc	17 128 30                       ; 3rdparty/../src/tl_templates/hip/copy.h:128:30
	v_or_b32_e32 v173, v136, v195	;.loc	4 53 365                        ; /tmp/tmp_67acr4h.cpp:53:365
	v_add_u32_e32 v174, v195, v138	;.loc	4 0 365 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:0:365
	s_nop 0	;.loc	17 132 5 is_stmt 1              ; 3rdparty/../src/tl_templates/hip/copy.h:132:5
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_lshlrev_b32_e32 v165, 1, v170	;.loc	4 48 81                         ; /tmp/tmp_67acr4h.cpp:48:81
	v_lshlrev_b32_e32 v167, 1, v171
	v_lshl_or_b32 v169, v172, 1, v145	;.loc	4 53 81                         ; /tmp/tmp_67acr4h.cpp:53:81
	v_lshl_or_b32 v170, v173, 1, v145
	v_lshl_add_u32 v186, v174, 1, v145
	s_waitcnt vmcnt(8)	;.loc	17 63 3                         ; 3rdparty/../src/tl_templates/hip/copy.h:63:3
	s_barrier	;.loc	4 45 5                          ; /tmp/tmp_67acr4h.cpp:45:5
	ds_read_b128 v146 v147 v148 v149, v169	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	ds_read_b128 v150 v151 v152 v153, v165	;.loc	4 48 60                         ; /tmp/tmp_67acr4h.cpp:48:60
	ds_read_b128 v154 v155 v156 v157, v186 offset:2048	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	ds_read_b128 v158 v159 v160 v161, v167	;.loc	4 48 60                         ; /tmp/tmp_67acr4h.cpp:48:60
	ds_read_b128 v162 v163 v164 v165, v170	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	ds_read_b128 v166 v167 v168 v169, v186 offset:4096
	ds_read_b128 v170 v171 v172 v173, v186 offset:6144
	ds_read_b128 v174 v175 v176 v177, v186 offset:8192
	ds_read_b128 v178 v179 v180 v181, v186 offset:10240
	ds_read_b128 v182 v183 v184 v185, v186 offset:12288
	ds_read_b128 v186 v187 v188 v189, v186 offset:14336
	v_add_lshl_u32 v196, v194, v138, 1	;.loc	4 48 81                         ; /tmp/tmp_67acr4h.cpp:48:81
	s_waitcnt lgkmcnt(9)	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v146 v147 v148 v149, v150 v151 v152 v153, v126 v127 v128 v129
	s_add_i32 s17, s17, 64	;.loc	4 32 21                         ; /tmp/tmp_67acr4h.cpp:32:21
	s_addk_i32 s16, 0x4000
	s_cmpk_eq_i32 s17, 0x1fc0
	s_waitcnt lgkmcnt(8)	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v154 v155 v156 v157, v150 v151 v152 v153, v122 v123 v124 v125
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v166 v167 v168 v169, v150 v151 v152 v153, v118 v119 v120 v121
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v170 v171 v172 v173, v150 v151 v152 v153, v114 v115 v116 v117
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v174 v175 v176 v177, v150 v151 v152 v153, v110 v111 v112 v113
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v178 v179 v180 v181, v150 v151 v152 v153, v106 v107 v108 v109
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v182 v183 v184 v185, v150 v151 v152 v153, v102 v103 v104 v105
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v186 v187 v188 v189, v150 v151 v152 v153, v098 v099 v100 v101
	ds_read_b128 v150 v151 v152 v153, v196 offset:2048	;.loc	4 48 60                         ; /tmp/tmp_67acr4h.cpp:48:60
	ds_read_b128 v190 v191 v192 v193, v196 offset:4096
	s_waitcnt lgkmcnt(1)	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v146 v147 v148 v149, v150 v151 v152 v153, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v154 v155 v156 v157, v150 v151 v152 v153, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v166 v167 v168 v169, v150 v151 v152 v153, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v170 v171 v172 v173, v150 v151 v152 v153, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v174 v175 v176 v177, v150 v151 v152 v153, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v178 v179 v180 v181, v150 v151 v152 v153, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v182 v183 v184 v185, v150 v151 v152 v153, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v186 v187 v188 v189, v150 v151 v152 v153, v062 v063 v064 v065
	ds_read_b128 v150 v151 v152 v153, v196 offset:6144	;.loc	4 48 60                         ; /tmp/tmp_67acr4h.cpp:48:60
	s_waitcnt lgkmcnt(1)	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v146 v147 v148 v149, v190 v191 v192 v193, v058 v059 v060 v061
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v146 v147 v148 v149, v150 v151 v152 v153, v026 v027 v028 v029
	v_add_u32_e32 v146, v195, v136
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v178 v179 v180 v181, v190 v191 v192 v193, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v178 v179 v180 v181, v150 v151 v152 v153, v006 v007 v008 v009
	v_lshl_add_u32 v178, v146, 1, v145	;.loc	4 53 81                         ; /tmp/tmp_67acr4h.cpp:53:81
	ds_read_b128 v146 v147 v148 v149, v178 offset:2048	;.loc	4 53 60 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:53:60
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v154 v155 v156 v157, v190 v191 v192 v193, v054 v055 v056 v057	;.loc	4 60 54 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:60:54
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v166 v167 v168 v169, v190 v191 v192 v193, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v170 v171 v172 v173, v190 v191 v192 v193, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v174 v175 v176 v177, v190 v191 v192 v193, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v154 v155 v156 v157, v150 v151 v152 v153, v022 v023 v024 v025
	ds_read_b128 v154 v155 v156 v157, v178 offset:6144	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v166 v167 v168 v169, v150 v151 v152 v153, v018 v019 v020 v021	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	ds_read_b128 v166 v167 v168 v169, v178 offset:8192	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v170 v171 v172 v173, v150 v151 v152 v153, v014 v015 v016 v017	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	ds_read_b128 v170 v171 v172 v173, v178 offset:10240	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v174 v175 v176 v177, v150 v151 v152 v153, v010 v011 v012 v013	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	ds_read_b128 v174 v175 v176 v177, v178 offset:12288	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v182 v183 v184 v185, v150 v151 v152 v153, v002 v003 v004 v005	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v186 v187 v188 v189, v150 v151 v152 v153, v094 v095 v096 v097
	ds_read_b128 v150 v151 v152 v153, v178 offset:4096	;.loc	4 53 60                         ; /tmp/tmp_67acr4h.cpp:53:60
	ds_read_b128 v178 v179 v180 v181, v178 offset:14336
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v182 v183 v184 v185, v190 v191 v192 v193, v034 v035 v036 v037	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	v_add_lshl_u32 v182, v194, v136, 1	;.loc	4 48 81                         ; /tmp/tmp_67acr4h.cpp:48:81
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v162 v163 v164 v165, v158 v159 v160 v161, v126 v127 v128 v129	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v146 v147 v148 v149, v158 v159 v160 v161, v122 v123 v124 v125
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v150 v151 v152 v153, v158 v159 v160 v161, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v154 v155 v156 v157, v158 v159 v160 v161, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v166 v167 v168 v169, v158 v159 v160 v161, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v170 v171 v172 v173, v158 v159 v160 v161, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v174 v175 v176 v177, v158 v159 v160 v161, v102 v103 v104 v105
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v178 v179 v180 v181, v158 v159 v160 v161, v098 v099 v100 v101
	ds_read_b128 v158 v159 v160 v161, v182 offset:2048	;.loc	4 48 60                         ; /tmp/tmp_67acr4h.cpp:48:60
	s_waitcnt lgkmcnt(0)	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v162 v163 v164 v165, v158 v159 v160 v161, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v146 v147 v148 v149, v158 v159 v160 v161, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v150 v151 v152 v153, v158 v159 v160 v161, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v154 v155 v156 v157, v158 v159 v160 v161, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v166 v167 v168 v169, v158 v159 v160 v161, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v170 v171 v172 v173, v158 v159 v160 v161, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v174 v175 v176 v177, v158 v159 v160 v161, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v178 v179 v180 v181, v158 v159 v160 v161, v062 v063 v064 v065
	ds_read_b128 v158 v159 v160 v161, v182 offset:4096	;.loc	4 48 60                         ; /tmp/tmp_67acr4h.cpp:48:60
	ds_read_b128 v182 v183 v184 v185, v182 offset:6144
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v186 v187 v188 v189, v190 v191 v192 v193, v030 v031 v032 v033	;.loc	4 60 54                         ; /tmp/tmp_67acr4h.cpp:60:54
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v162 v163 v164 v165, v158 v159 v160 v161, v058 v059 v060 v061
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v146 v147 v148 v149, v158 v159 v160 v161, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v150 v151 v152 v153, v158 v159 v160 v161, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v154 v155 v156 v157, v158 v159 v160 v161, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v166 v167 v168 v169, v158 v159 v160 v161, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v170 v171 v172 v173, v158 v159 v160 v161, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v174 v175 v176 v177, v158 v159 v160 v161, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v178 v179 v180 v181, v158 v159 v160 v161, v030 v031 v032 v033
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v162 v163 v164 v165, v182 v183 v184 v185, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v146 v147 v148 v149, v182 v183 v184 v185, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v150 v151 v152 v153, v182 v183 v184 v185, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v154 v155 v156 v157, v182 v183 v184 v185, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v166 v167 v168 v169, v182 v183 v184 v185, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v170 v171 v172 v173, v182 v183 v184 v185, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v174 v175 v176 v177, v182 v183 v184 v185, v002 v003 v004 v005
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v178 v179 v180 v181, v182 v183 v184 v185, v094 v095 v096 v097
.JUMP.LBB0_3:
	s_cbranch_scc0 .LBB0_3	;.loc	4 32 3                          ; /tmp/tmp_67acr4h.cpp:32:3
	v_and_b32_e32 v001, 0x33c0, v139	;.loc	4 0 3 is_stmt 0                 ; /tmp/tmp_67acr4h.cpp:0:3
	v_or3_b32 v001, v001, v134, v135	;.loc	4 70 8 is_stmt 1                ; /tmp/tmp_67acr4h.cpp:70:8
	v_or_b32_e32 v130, v138, v001	;.loc	4 72 354                        ; /tmp/tmp_67acr4h.cpp:72:354
	v_lshlrev_b32_e32 v134, 1, v130	;.loc	4 72 81 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:72:81
	v_or_b32_e32 v130, v136, v001	;.loc	4 72 354                        ; /tmp/tmp_67acr4h.cpp:72:354
	v_lshlrev_b32_e32 v135, 1, v130	;.loc	4 72 81                         ; /tmp/tmp_67acr4h.cpp:72:81
	v_or_b32_e32 v130, v138, v137	;.loc	4 77 346 is_stmt 1              ; /tmp/tmp_67acr4h.cpp:77:346
	v_mov_b32_e32 v184, 0x18000	;.loc	4 77 81 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:77:81
	v_lshl_or_b32 v130, v130, 1, v184
	s_waitcnt vmcnt(0)	;.loc	17 63 3 is_stmt 1               ; 3rdparty/../src/tl_templates/hip/copy.h:63:3
	s_barrier	;.loc	4 69 3                          ; /tmp/tmp_67acr4h.cpp:69:3
	ds_read_b128 v130 v131 v132 v133, v130	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	ds_read_b128 v140 v141 v142 v143, v134 offset:32768	;.loc	4 72 60                         ; /tmp/tmp_67acr4h.cpp:72:60
	ds_read_b128 v144 v145 v146 v147, v135 offset:32768
	v_add_u32_e32 v135, v138, v137	;.loc	4 0 60 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:60
	v_lshl_add_u32 v135, v135, 1, v184	;.loc	4 77 81 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:77:81
	ds_read_b128 v148 v149 v150 v151, v135 offset:2048	;.loc	4 77 60 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:77:60
	ds_read_b128 v156 v157 v158 v159, v135 offset:4096
	ds_read_b128 v160 v161 v162 v163, v135 offset:6144
	ds_read_b128 v164 v165 v166 v167, v135 offset:8192
	ds_read_b128 v168 v169 v170 v171, v135 offset:10240
	ds_read_b128 v172 v173 v174 v175, v135 offset:12288
	ds_read_b128 v176 v177 v178 v179, v135 offset:14336
	v_or_b32_e32 v134, v136, v137	;.loc	4 77 346                        ; /tmp/tmp_67acr4h.cpp:77:346
	v_lshl_or_b32 v134, v134, 1, v184	;.loc	4 77 81                         ; /tmp/tmp_67acr4h.cpp:77:81
	ds_read_b128 v152 v153 v154 v155, v134	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	v_add_lshl_u32 v134, v138, v001, 1	;.loc	4 72 81 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:72:81
	s_waitcnt lgkmcnt(9)	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v130 v131 v132 v133, v140 v141 v142 v143, v126 v127 v128 v129
	v_add_lshl_u32 v001, v136, v001, 1	;.loc	4 72 81                         ; /tmp/tmp_67acr4h.cpp:72:81
	s_mov_b32 s0, 0x7f800000
	s_waitcnt lgkmcnt(7)	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v148 v149 v150 v151, v140 v141 v142 v143, v122 v123 v124 v125
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v156 v157 v158 v159, v140 v141 v142 v143, v118 v119 v120 v121
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v160 v161 v162 v163, v140 v141 v142 v143, v114 v115 v116 v117
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v164 v165 v166 v167, v140 v141 v142 v143, v110 v111 v112 v113
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v168 v169 v170 v171, v140 v141 v142 v143, v106 v107 v108 v109
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v172 v173 v174 v175, v140 v141 v142 v143, v102 v103 v104 v105
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v176 v177 v178 v179, v140 v141 v142 v143, v098 v099 v100 v101
	ds_read_b128 v138 v139 v140 v141, v134 offset:34816	;.loc	4 72 60                         ; /tmp/tmp_67acr4h.cpp:72:60
	ds_read_b128 v180 v181 v182 v183, v134 offset:36864
	s_waitcnt lgkmcnt(1)	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v130 v131 v132 v133, v138 v139 v140 v141, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v148 v149 v150 v151, v138 v139 v140 v141, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v156 v157 v158 v159, v138 v139 v140 v141, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v160 v161 v162 v163, v138 v139 v140 v141, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v164 v165 v166 v167, v138 v139 v140 v141, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v168 v169 v170 v171, v138 v139 v140 v141, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v172 v173 v174 v175, v138 v139 v140 v141, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v176 v177 v178 v179, v138 v139 v140 v141, v062 v063 v064 v065
	ds_read_b128 v138 v139 v140 v141, v134 offset:38912	;.loc	4 72 60                         ; /tmp/tmp_67acr4h.cpp:72:60
	s_waitcnt lgkmcnt(1)	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v130 v131 v132 v133, v180 v181 v182 v183, v058 v059 v060 v061
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v130 v131 v132 v133, v138 v139 v140 v141, v026 v027 v028 v029
	v_add_u32_e32 v130, v136, v137	;.loc	4 0 54 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:54
	v_lshl_add_u32 v134, v130, 1, v184	;.loc	4 77 81 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:77:81
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v148 v149 v150 v151, v180 v181 v182 v183, v054 v055 v056 v057	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v156 v157 v158 v159, v180 v181 v182 v183, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v160 v161 v162 v163, v180 v181 v182 v183, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v164 v165 v166 v167, v180 v181 v182 v183, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v168 v169 v170 v171, v180 v181 v182 v183, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v172 v173 v174 v175, v180 v181 v182 v183, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v148 v149 v150 v151, v138 v139 v140 v141, v022 v023 v024 v025
	ds_read_b128 v148 v149 v150 v151, v134 offset:4096	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v156 v157 v158 v159, v138 v139 v140 v141, v018 v019 v020 v021	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	ds_read_b128 v156 v157 v158 v159, v134 offset:6144	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v160 v161 v162 v163, v138 v139 v140 v141, v014 v015 v016 v017	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	ds_read_b128 v160 v161 v162 v163, v134 offset:8192	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v164 v165 v166 v167, v138 v139 v140 v141, v010 v011 v012 v013	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	ds_read_b128 v164 v165 v166 v167, v134 offset:10240	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v168 v169 v170 v171, v138 v139 v140 v141, v006 v007 v008 v009	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	ds_read_b128 v168 v169 v170 v171, v134 offset:12288	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v172 v173 v174 v175, v138 v139 v140 v141, v002 v003 v004 v005	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	ds_read_b128 v172 v173 v174 v175, v134 offset:14336	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	v_mfma_f32_16x16x32_bf16 v130 v131 v132 v133, v176 v177 v178 v179, v138 v139 v140 v141, v094 v095 v096 v097	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	ds_read_b128 v138 v139 v140 v141, v134 offset:2048	;.loc	4 77 60                         ; /tmp/tmp_67acr4h.cpp:77:60
	ds_read_b128 v134 v135 v136 v137, v001 offset:34816	;.loc	4 72 60                         ; /tmp/tmp_67acr4h.cpp:72:60
	v_mfma_f32_16x16x32_bf16 v126 v127 v128 v129, v152 v153 v154 v155, v144 v145 v146 v147, v126 v127 v128 v129	;.loc	4 84 54                         ; /tmp/tmp_67acr4h.cpp:84:54
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v122 v123 v124 v125, v138 v139 v140 v141, v144 v145 v146 v147, v122 v123 v124 v125
	v_mfma_f32_16x16x32_bf16 v118 v119 v120 v121, v148 v149 v150 v151, v144 v145 v146 v147, v118 v119 v120 v121
	v_mfma_f32_16x16x32_bf16 v114 v115 v116 v117, v156 v157 v158 v159, v144 v145 v146 v147, v114 v115 v116 v117
	v_mfma_f32_16x16x32_bf16 v110 v111 v112 v113, v160 v161 v162 v163, v144 v145 v146 v147, v110 v111 v112 v113
	v_mfma_f32_16x16x32_bf16 v106 v107 v108 v109, v164 v165 v166 v167, v144 v145 v146 v147, v106 v107 v108 v109
	v_mfma_f32_16x16x32_bf16 v102 v103 v104 v105, v168 v169 v170 v171, v144 v145 v146 v147, v102 v103 v104 v105
	v_mfma_f32_16x16x32_bf16 v098 v099 v100 v101, v172 v173 v174 v175, v144 v145 v146 v147, v098 v099 v100 v101
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v094 v095 v096 v097, v152 v153 v154 v155, v134 v135 v136 v137, v090 v091 v092 v093
	v_mfma_f32_16x16x32_bf16 v090 v091 v092 v093, v138 v139 v140 v141, v134 v135 v136 v137, v086 v087 v088 v089
	v_mfma_f32_16x16x32_bf16 v086 v087 v088 v089, v148 v149 v150 v151, v134 v135 v136 v137, v082 v083 v084 v085
	v_mfma_f32_16x16x32_bf16 v082 v083 v084 v085, v156 v157 v158 v159, v134 v135 v136 v137, v078 v079 v080 v081
	v_mfma_f32_16x16x32_bf16 v078 v079 v080 v081, v160 v161 v162 v163, v134 v135 v136 v137, v074 v075 v076 v077
	v_mfma_f32_16x16x32_bf16 v074 v075 v076 v077, v164 v165 v166 v167, v134 v135 v136 v137, v070 v071 v072 v073
	v_mfma_f32_16x16x32_bf16 v070 v071 v072 v073, v168 v169 v170 v171, v134 v135 v136 v137, v066 v067 v068 v069
	v_mfma_f32_16x16x32_bf16 v066 v067 v068 v069, v172 v173 v174 v175, v134 v135 v136 v137, v062 v063 v064 v065
	ds_read_b128 v134 v135 v136 v137, v001 offset:36864	;.loc	4 72 60                         ; /tmp/tmp_67acr4h.cpp:72:60
	ds_read_b128 v142 v143 v144 v145, v001 offset:38912
	v_and_b32_e32 v001, 0x7f800000, v126	;.loc	2 101 18                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v176 v177 v178 v179, v180 v181 v182 v183, v030 v031 v032 v033	;.loc	4 84 54 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:84:54
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v062 v063 v064 v065, v152 v153 v154 v155, v134 v135 v136 v137, v058 v059 v060 v061
	v_mfma_f32_16x16x32_bf16 v058 v059 v060 v061, v138 v139 v140 v141, v134 v135 v136 v137, v054 v055 v056 v057
	v_mfma_f32_16x16x32_bf16 v054 v055 v056 v057, v148 v149 v150 v151, v134 v135 v136 v137, v050 v051 v052 v053
	v_mfma_f32_16x16x32_bf16 v050 v051 v052 v053, v156 v157 v158 v159, v134 v135 v136 v137, v046 v047 v048 v049
	v_mfma_f32_16x16x32_bf16 v046 v047 v048 v049, v160 v161 v162 v163, v134 v135 v136 v137, v042 v043 v044 v045
	v_mfma_f32_16x16x32_bf16 v042 v043 v044 v045, v164 v165 v166 v167, v134 v135 v136 v137, v038 v039 v040 v041
	v_mfma_f32_16x16x32_bf16 v038 v039 v040 v041, v168 v169 v170 v171, v134 v135 v136 v137, v034 v035 v036 v037
	v_mfma_f32_16x16x32_bf16 v034 v035 v036 v037, v172 v173 v174 v175, v134 v135 v136 v137, v030 v031 v032 v033
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v030 v031 v032 v033, v152 v153 v154 v155, v142 v143 v144 v145, v026 v027 v028 v029
	v_mfma_f32_16x16x32_bf16 v026 v027 v028 v029, v138 v139 v140 v141, v142 v143 v144 v145, v022 v023 v024 v025
	v_mfma_f32_16x16x32_bf16 v022 v023 v024 v025, v148 v149 v150 v151, v142 v143 v144 v145, v018 v019 v020 v021
	v_mfma_f32_16x16x32_bf16 v018 v019 v020 v021, v156 v157 v158 v159, v142 v143 v144 v145, v014 v015 v016 v017
	v_mfma_f32_16x16x32_bf16 v014 v015 v016 v017, v160 v161 v162 v163, v142 v143 v144 v145, v010 v011 v012 v013
	v_mfma_f32_16x16x32_bf16 v010 v011 v012 v013, v164 v165 v166 v167, v142 v143 v144 v145, v006 v007 v008 v009
	v_mfma_f32_16x16x32_bf16 v006 v007 v008 v009, v168 v169 v170 v171, v142 v143 v144 v145, v002 v003 v004 v005
	v_mfma_f32_16x16x32_bf16 v002 v003 v004 v005, v172 v173 v174 v175, v142 v143 v144 v145, v130 v131 v132 v133
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v126, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v126, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v130, 0x10000, v126	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v126, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v130, v126, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v126, 0x7f800000, v127	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v126	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v126, v127, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v130, v127, v126, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v126, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v130, 0x10000, v127
	v_cmp_eq_u32_sdwa vcc, v127, v126 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v130, v130, v127, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v126, 0x7f800000, v128	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v126	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v126, v128, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v131, v128, v126, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v126, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v127, 0x10000, v128
	v_cmp_eq_u32_sdwa vcc, v128, v126 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v131, v127, v128, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v126, 0x7f800000, v129	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v126	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v126, v129, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v132, v129, v126, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v126, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v127, 0x10000, v129
	v_cmp_eq_u32_sdwa vcc, v129, v126 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v132, v127, v129, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshlrev_b32_e32 v126, 13, v000
	v_lshrrev_b32_e32 v127, 1, v000
	v_lshrrev_b32_e32 v000, 2, v000
	s_mov_b32 s0, 0x19e080
	v_and_b32_e32 v000, 12, v000
	v_bitop3_b32 v126, v126, s0, v127 bitop3:0xc8
	v_or3_b32 v000, v126, v000, s14
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	v_lshl_add_u32 v000, s15, 8, v000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v126, 0xffff0000, v132	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_and_or_b32 v128, v130, s0, v001
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	4 100 17                        ; /tmp/tmp_67acr4h.cpp:100:17
	v_or_b32_sdwa v129, v126, v131 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_lshl_add_u64 v126 v127, v000 v001, 1, s[8:9]	;.loc	4 100 17                        ; /tmp/tmp_67acr4h.cpp:100:17
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v001, 0x7f800000, v122	;.loc	2 101 18                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v128 v129, off	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v122, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v122, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v128, 0x10000, v122	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v122, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v128, v122, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v122, 0x7f800000, v123	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v122	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v122, v123, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v128, v123, v122, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v122, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v128, 0x10000, v123
	v_cmp_eq_u32_sdwa vcc, v123, v122 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v128, v128, v123, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v122, 0x7f800000, v124	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v122	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v122, v124, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v129, v124, v122, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v122, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v123, 0x10000, v124
	v_cmp_eq_u32_sdwa vcc, v124, v122 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v129, v123, v124, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v122, 0x7f800000, v125	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v122	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v122, v125, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v130, v125, v122, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v122, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v123, 0x10000, v125
	v_cmp_eq_u32_sdwa vcc, v125, v122 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v130, v123, v125, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v122, 0xffff0000, v130	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v123, v122, v129 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v122, v128, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v118	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v122 v123, off offset:32	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v118, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v118, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v122, 0x10000, v118	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v118, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v122, v118, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v118, 0x7f800000, v119	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v118	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v118, v119, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v122, v119, v118, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v118, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v122, 0x10000, v119
	v_cmp_eq_u32_sdwa vcc, v119, v118 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v122, v122, v119, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v118, 0x7f800000, v120	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v118	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v118, v120, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v123, v120, v118, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v118, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v119, 0x10000, v120
	v_cmp_eq_u32_sdwa vcc, v120, v118 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v123, v119, v120, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v118, 0x7f800000, v121	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v118	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v118, v121, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v124, v121, v118, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v118, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v119, 0x10000, v121
	v_cmp_eq_u32_sdwa vcc, v121, v118 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v124, v119, v121, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v118, 0xffff0000, v124	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v119, v118, v123 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v118, v122, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v114	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v118 v119, off offset:64	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v114, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v114, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v118, 0x10000, v114	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v114, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v118, v114, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v114, 0x7f800000, v115	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v114	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v114, v115, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v118, v115, v114, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v114, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v118, 0x10000, v115
	v_cmp_eq_u32_sdwa vcc, v115, v114 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v118, v118, v115, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v114, 0x7f800000, v116	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v114	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v114, v116, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v119, v116, v114, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v114, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v115, 0x10000, v116
	v_cmp_eq_u32_sdwa vcc, v116, v114 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v119, v115, v116, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v114, 0x7f800000, v117	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v114	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v114, v117, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v120, v117, v114, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v114, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v115, 0x10000, v117
	v_cmp_eq_u32_sdwa vcc, v117, v114 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v120, v115, v117, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v114, 0xffff0000, v120	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v115, v114, v119 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v114, v118, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v110	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v114 v115, off offset:96	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v110, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v110, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v114, 0x10000, v110	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v110, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v114, v110, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v110, 0x7f800000, v111	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v110	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v110, v111, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v114, v111, v110, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v110, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v114, 0x10000, v111
	v_cmp_eq_u32_sdwa vcc, v111, v110 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v114, v114, v111, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v110, 0x7f800000, v112	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v110	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v110, v112, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v115, v112, v110, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v110, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v111, 0x10000, v112
	v_cmp_eq_u32_sdwa vcc, v112, v110 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v115, v111, v112, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v110, 0x7f800000, v113	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v110	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v110, v113, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v113, v110, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v110, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v111, 0x10000, v113
	v_cmp_eq_u32_sdwa vcc, v113, v110 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v116, v111, v113, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v110, 0xffff0000, v116	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v111, v110, v115 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v110, v114, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v106	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v110 v111, off offset:128	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v106, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v106, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v110, 0x10000, v106	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v106, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v110, v106, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v106, 0x7f800000, v107	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v106	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v106, v107, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v110, v107, v106, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v106, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v110, 0x10000, v107
	v_cmp_eq_u32_sdwa vcc, v107, v106 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v110, v110, v107, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v106, 0x7f800000, v108	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v106	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v106, v108, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v111, v108, v106, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v106, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v107, 0x10000, v108
	v_cmp_eq_u32_sdwa vcc, v108, v106 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v111, v107, v108, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v106, 0x7f800000, v109	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v106	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v106, v109, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v112, v109, v106, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v106, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v107, 0x10000, v109
	v_cmp_eq_u32_sdwa vcc, v109, v106 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v112, v107, v109, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v106, 0xffff0000, v112	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v107, v106, v111 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v106, v110, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v102	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v106 v107, off offset:160	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v102, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v102, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v106, 0x10000, v102	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v102, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v106, v102, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v102, 0x7f800000, v103	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v102	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v102, v103, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v106, v103, v102, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v102, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v106, 0x10000, v103
	v_cmp_eq_u32_sdwa vcc, v103, v102 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v106, v106, v103, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v102, 0x7f800000, v104	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v102	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v102, v104, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v107, v104, v102, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v102, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v103, 0x10000, v104
	v_cmp_eq_u32_sdwa vcc, v104, v102 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v107, v103, v104, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v102, 0x7f800000, v105	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v102	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v102, v105, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v108, v105, v102, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v102, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v103, 0x10000, v105
	v_cmp_eq_u32_sdwa vcc, v105, v102 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v108, v103, v105, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v102, 0xffff0000, v108	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v103, v102, v107 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v102, v106, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v098	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v102 v103, off offset:192	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v098, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v098, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v102, 0x10000, v098	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v098, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v102, v098, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v098, 0x7f800000, v099	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v098	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v098, v099, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v102, v099, v098, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v098, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v102, 0x10000, v099
	v_cmp_eq_u32_sdwa vcc, v099, v098 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v102, v102, v099, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v098, 0x7f800000, v100	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v098	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v098, v100, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v103, v100, v098, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v098, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v099, 0x10000, v100
	v_cmp_eq_u32_sdwa vcc, v100, v098 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v103, v099, v100, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v098, 0x7f800000, v101	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v098	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v098, v101, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v104, v101, v098, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v098, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v099, 0x10000, v101
	v_cmp_eq_u32_sdwa vcc, v101, v098 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v104, v099, v101, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v098, 0xffff0000, v104	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v099, v098, v103 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v098, v102, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v094	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v126 v127, v098 v099, off offset:224	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v094, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v094, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v098, 0x10000, v094	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v094, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v098, v094, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v094, 0x7f800000, v095	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v094	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v094, v095, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v098, v095, v094, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v094, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v098, 0x10000, v095
	v_cmp_eq_u32_sdwa vcc, v095, v094 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v098, v098, v095, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v094, 0x7f800000, v096	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v094	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v094, v096, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v099, v096, v094, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v094, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v095, 0x10000, v096
	v_cmp_eq_u32_sdwa vcc, v096, v094 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v099, v095, v096, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v094, 0x7f800000, v097	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v094	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v094, v097, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v100, v097, v094, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v094, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v095, 0x10000, v097
	v_cmp_eq_u32_sdwa vcc, v097, v094 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v100, v095, v097, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_and_b32_e32 v094, 0xffff0000, v100	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_or_b32_sdwa v097, v094, v099 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_add_u32_e32 v094, 0x20000, v000	;.loc	4 100 196                       ; /tmp/tmp_67acr4h.cpp:100:196
	v_and_or_b32 v096, v098, s0, v001	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_ashrrev_i32_e32 v095, 31, v094	;.loc	4 100 17                        ; /tmp/tmp_67acr4h.cpp:100:17
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v001, 0x7f800000, v090	;.loc	2 101 18                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_lshl_add_u64 v094 v095, v094 v095, 1, s[8:9]	;.loc	4 100 17                        ; /tmp/tmp_67acr4h.cpp:100:17
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v096 v097, off	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v090, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v090, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v096, 0x10000, v090	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v090, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v096, v090, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v090, 0x7f800000, v091	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v090	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v090, v091, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v096, v091, v090, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v090, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v096, 0x10000, v091
	v_cmp_eq_u32_sdwa vcc, v091, v090 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v096, v096, v091, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v090, 0x7f800000, v092	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v090	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v090, v092, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v097, v092, v090, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v090, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v091, 0x10000, v092
	v_cmp_eq_u32_sdwa vcc, v092, v090 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v097, v091, v092, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v090, 0x7f800000, v093	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v090	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v090, v093, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v098, v093, v090, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v090, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v091, 0x10000, v093
	v_cmp_eq_u32_sdwa vcc, v093, v090 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v098, v091, v093, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v090, 0xffff0000, v098	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v091, v090, v097 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v090, v096, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v086	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v090 v091, off offset:32	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v086, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v086, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v090, 0x10000, v086	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v086, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v090, v086, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v086, 0x7f800000, v087	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v086	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v086, v087, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v090, v087, v086, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v086, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v090, 0x10000, v087
	v_cmp_eq_u32_sdwa vcc, v087, v086 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v090, v090, v087, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v086, 0x7f800000, v088	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v086	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v086, v088, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v091, v088, v086, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v086, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v087, 0x10000, v088
	v_cmp_eq_u32_sdwa vcc, v088, v086 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v091, v087, v088, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v086, 0x7f800000, v089	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v086	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v086, v089, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v092, v089, v086, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v086, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v087, 0x10000, v089
	v_cmp_eq_u32_sdwa vcc, v089, v086 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v092, v087, v089, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v086, 0xffff0000, v092	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v087, v086, v091 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v086, v090, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v082	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v086 v087, off offset:64	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v082, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v082, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v086, 0x10000, v082	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v082, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v086, v082, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v082, 0x7f800000, v083	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v082	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v082, v083, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v086, v083, v082, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v082, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v086, 0x10000, v083
	v_cmp_eq_u32_sdwa vcc, v083, v082 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v086, v086, v083, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v082, 0x7f800000, v084	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v082	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v082, v084, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v087, v084, v082, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v082, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v083, 0x10000, v084
	v_cmp_eq_u32_sdwa vcc, v084, v082 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v087, v083, v084, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v082, 0x7f800000, v085	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v082	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v082, v085, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v088, v085, v082, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v082, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v083, 0x10000, v085
	v_cmp_eq_u32_sdwa vcc, v085, v082 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v088, v083, v085, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v082, 0xffff0000, v088	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v083, v082, v087 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v082, v086, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v078	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v082 v083, off offset:96	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v078, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v078, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v082, 0x10000, v078	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v078, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v082, v078, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v078, 0x7f800000, v079	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v078	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v078, v079, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v082, v079, v078, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v078, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v082, 0x10000, v079
	v_cmp_eq_u32_sdwa vcc, v079, v078 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v082, v082, v079, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v078, 0x7f800000, v080	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v078	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v078, v080, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v083, v080, v078, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v078, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v079, 0x10000, v080
	v_cmp_eq_u32_sdwa vcc, v080, v078 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v083, v079, v080, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v078, 0x7f800000, v081	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v078	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v078, v081, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v084, v081, v078, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v078, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v079, 0x10000, v081
	v_cmp_eq_u32_sdwa vcc, v081, v078 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v084, v079, v081, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v078, 0xffff0000, v084	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v079, v078, v083 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v078, v082, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v074	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v078 v079, off offset:128	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v074, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v074, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v078, 0x10000, v074	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v074, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v078, v074, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v074, 0x7f800000, v075	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v074	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v074, v075, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v078, v075, v074, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v074, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v078, 0x10000, v075
	v_cmp_eq_u32_sdwa vcc, v075, v074 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v078, v078, v075, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v074, 0x7f800000, v076	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v074	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v074, v076, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v079, v076, v074, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v074, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v075, 0x10000, v076
	v_cmp_eq_u32_sdwa vcc, v076, v074 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v079, v075, v076, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v074, 0x7f800000, v077	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v074	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v074, v077, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v080, v077, v074, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v074, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v075, 0x10000, v077
	v_cmp_eq_u32_sdwa vcc, v077, v074 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v080, v075, v077, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v074, 0xffff0000, v080	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v075, v074, v079 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v074, v078, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v070	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v074 v075, off offset:160	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v070, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v070, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v074, 0x10000, v070	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v070, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v074, v070, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v070, 0x7f800000, v071	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v070	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v070, v071, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v074, v071, v070, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v070, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v074, 0x10000, v071
	v_cmp_eq_u32_sdwa vcc, v071, v070 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v074, v074, v071, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v070, 0x7f800000, v072	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v070	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v070, v072, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v075, v072, v070, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v070, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v071, 0x10000, v072
	v_cmp_eq_u32_sdwa vcc, v072, v070 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v075, v071, v072, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v070, 0x7f800000, v073	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v070	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v070, v073, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v076, v073, v070, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v070, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v071, 0x10000, v073
	v_cmp_eq_u32_sdwa vcc, v073, v070 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v076, v071, v073, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v070, 0xffff0000, v076	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v071, v070, v075 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v070, v074, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v066	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v070 v071, off offset:192	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v066, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v066, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v070, 0x10000, v066	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v066, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v070, v066, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v066, 0x7f800000, v067	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v066	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v066, v067, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v070, v067, v066, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v066, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v070, 0x10000, v067
	v_cmp_eq_u32_sdwa vcc, v067, v066 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v070, v070, v067, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v066, 0x7f800000, v068	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v066	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v066, v068, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v071, v068, v066, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v066, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v067, 0x10000, v068
	v_cmp_eq_u32_sdwa vcc, v068, v066 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v071, v067, v068, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v066, 0x7f800000, v069	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v066	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v066, v069, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v072, v069, v066, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v066, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v067, 0x10000, v069
	v_cmp_eq_u32_sdwa vcc, v069, v066 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v072, v067, v069, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v066, 0xffff0000, v072	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v067, v066, v071 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v066, v070, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v062	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v094 v095, v066 v067, off offset:224	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v062, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v062, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v066, 0x10000, v062	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v062, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v066, v062, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v062, 0x7f800000, v063	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v062	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v062, v063, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v066, v063, v062, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v062, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v066, 0x10000, v063
	v_cmp_eq_u32_sdwa vcc, v063, v062 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v066, v066, v063, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v062, 0x7f800000, v064	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v062	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v062, v064, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v067, v064, v062, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v062, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v063, 0x10000, v064
	v_cmp_eq_u32_sdwa vcc, v064, v062 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v067, v063, v064, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v062, 0x7f800000, v065	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v062	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v062, v065, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v068, v065, v062, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v062, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v063, 0x10000, v065
	v_cmp_eq_u32_sdwa vcc, v065, v062 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v068, v063, v065, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_and_b32_e32 v062, 0xffff0000, v068	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_or_b32_sdwa v065, v062, v067 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_add_u32_e32 v062, 0x40000, v000	;.loc	4 100 196                       ; /tmp/tmp_67acr4h.cpp:100:196
	v_and_or_b32 v064, v066, s0, v001	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_ashrrev_i32_e32 v063, 31, v062	;.loc	4 100 17                        ; /tmp/tmp_67acr4h.cpp:100:17
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v001, 0x7f800000, v058	;.loc	2 101 18                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_lshl_add_u64 v062 v063, v062 v063, 1, s[8:9]	;.loc	4 100 17                        ; /tmp/tmp_67acr4h.cpp:100:17
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v064 v065, off	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v058, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v058, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v064, 0x10000, v058	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v058, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v064, v058, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v058, 0x7f800000, v059	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v058	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v058, v059, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v064, v059, v058, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v058, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v064, 0x10000, v059
	v_cmp_eq_u32_sdwa vcc, v059, v058 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v064, v064, v059, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v058, 0x7f800000, v060	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v058	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v058, v060, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v065, v060, v058, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v058, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v059, 0x10000, v060
	v_cmp_eq_u32_sdwa vcc, v060, v058 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v065, v059, v060, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v058, 0x7f800000, v061	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v058	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v058, v061, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v066, v061, v058, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v058, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v059, 0x10000, v061
	v_cmp_eq_u32_sdwa vcc, v061, v058 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v066, v059, v061, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v058, 0xffff0000, v066	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v059, v058, v065 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v058, v064, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v054	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v058 v059, off offset:32	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v054, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v054, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v058, 0x10000, v054	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v054, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v058, v054, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v054, 0x7f800000, v055	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v054	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v054, v055, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v058, v055, v054, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v054, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v058, 0x10000, v055
	v_cmp_eq_u32_sdwa vcc, v055, v054 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v058, v058, v055, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v054, 0x7f800000, v056	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v054	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v054, v056, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v059, v056, v054, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v054, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v055, 0x10000, v056
	v_cmp_eq_u32_sdwa vcc, v056, v054 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v059, v055, v056, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v054, 0x7f800000, v057	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v054	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v054, v057, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v060, v057, v054, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v054, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v055, 0x10000, v057
	v_cmp_eq_u32_sdwa vcc, v057, v054 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v060, v055, v057, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v054, 0xffff0000, v060	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v055, v054, v059 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v054, v058, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v050	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v054 v055, off offset:64	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v050, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v050, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v054, 0x10000, v050	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v050, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v054, v050, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v050, 0x7f800000, v051	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v050	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v050, v051, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v054, v051, v050, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v050, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v054, 0x10000, v051
	v_cmp_eq_u32_sdwa vcc, v051, v050 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v054, v054, v051, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v050, 0x7f800000, v052	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v050	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v050, v052, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v055, v052, v050, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v050, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v051, 0x10000, v052
	v_cmp_eq_u32_sdwa vcc, v052, v050 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v055, v051, v052, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v050, 0x7f800000, v053	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v050	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v050, v053, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v056, v053, v050, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v050, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v051, 0x10000, v053
	v_cmp_eq_u32_sdwa vcc, v053, v050 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v056, v051, v053, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v050, 0xffff0000, v056	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v051, v050, v055 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v050, v054, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v046	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v050 v051, off offset:96	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v046, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v046, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v050, 0x10000, v046	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v046, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v050, v046, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v046, 0x7f800000, v047	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v046	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v046, v047, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v050, v047, v046, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v046, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v050, 0x10000, v047
	v_cmp_eq_u32_sdwa vcc, v047, v046 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v050, v050, v047, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v046, 0x7f800000, v048	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v046	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v046, v048, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v051, v048, v046, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v046, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v047, 0x10000, v048
	v_cmp_eq_u32_sdwa vcc, v048, v046 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v051, v047, v048, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v046, 0x7f800000, v049	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v046	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v046, v049, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v052, v049, v046, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v046, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v047, 0x10000, v049
	v_cmp_eq_u32_sdwa vcc, v049, v046 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v052, v047, v049, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v046, 0xffff0000, v052	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v047, v046, v051 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v046, v050, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v042	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v046 v047, off offset:128	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v042, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v042, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v046, 0x10000, v042	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v042, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v046, v042, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v042, 0x7f800000, v043	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v042	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v042, v043, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v046, v043, v042, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v042, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v046, 0x10000, v043
	v_cmp_eq_u32_sdwa vcc, v043, v042 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v046, v046, v043, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v042, 0x7f800000, v044	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v042	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v042, v044, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v047, v044, v042, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v042, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v043, 0x10000, v044
	v_cmp_eq_u32_sdwa vcc, v044, v042 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v047, v043, v044, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v042, 0x7f800000, v045	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v042	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v042, v045, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v048, v045, v042, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v042, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v043, 0x10000, v045
	v_cmp_eq_u32_sdwa vcc, v045, v042 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v048, v043, v045, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v042, 0xffff0000, v048	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v043, v042, v047 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v042, v046, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v038	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v042 v043, off offset:160	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v038, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v038, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v042, 0x10000, v038	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v038, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v042, v038, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v038, 0x7f800000, v039	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v038	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v038, v039, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v042, v039, v038, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v038, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v042, 0x10000, v039
	v_cmp_eq_u32_sdwa vcc, v039, v038 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v042, v042, v039, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v038, 0x7f800000, v040	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v038	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v038, v040, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v043, v040, v038, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v038, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v039, 0x10000, v040
	v_cmp_eq_u32_sdwa vcc, v040, v038 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v043, v039, v040, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v038, 0x7f800000, v041	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v038	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v038, v041, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v044, v041, v038, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v038, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v039, 0x10000, v041
	v_cmp_eq_u32_sdwa vcc, v041, v038 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v044, v039, v041, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v038, 0xffff0000, v044	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v039, v038, v043 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v038, v042, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v034	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v038 v039, off offset:192	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v034, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v034, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v038, 0x10000, v034	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v034, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v038, v034, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v034, 0x7f800000, v035	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v034	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v034, v035, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v038, v035, v034, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v034, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v038, 0x10000, v035
	v_cmp_eq_u32_sdwa vcc, v035, v034 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v038, v038, v035, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v034, 0x7f800000, v036	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v034	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v034, v036, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v039, v036, v034, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v034, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v035, 0x10000, v036
	v_cmp_eq_u32_sdwa vcc, v036, v034 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v039, v035, v036, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v034, 0x7f800000, v037	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v034	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v034, v037, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v040, v037, v034, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v034, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v035, 0x10000, v037
	v_cmp_eq_u32_sdwa vcc, v037, v034 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v040, v035, v037, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v034, 0xffff0000, v040	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v035, v034, v039 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v034, v038, s0, v001
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 33 is_stmt 0                ; /tmp/tmp_67acr4h.cpp:0:33
	v_and_b32_e32 v001, 0x7f800000, v030	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v001	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	global_store_dwordx2 v062 v063, v034 v035, off offset:224	;.loc	4 100 296 is_stmt 1             ; /tmp/tmp_67acr4h.cpp:100:296
	s_and_saveexec_b64 s[0:1], vcc	;.loc	2 101 9                         ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v001, v030, 16, 1	;.loc	2 118 44                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v001, v030, v001, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v001, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v034, 0x10000, v030	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v030, v001 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v001, v034, v030, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v030, 0x7f800000, v031	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v030	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v030, v031, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v034, v031, v030, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v030, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v034, 0x10000, v031
	v_cmp_eq_u32_sdwa vcc, v031, v030 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v034, v034, v031, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v030, 0x7f800000, v032	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v030	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v030, v032, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v035, v032, v030, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v030, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v031, 0x10000, v032
	v_cmp_eq_u32_sdwa vcc, v032, v030 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v035, v031, v032, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v030, 0x7f800000, v033	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v030	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v030, v033, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v036, v033, v030, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v030, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v031, 0x10000, v033
	v_cmp_eq_u32_sdwa vcc, v033, v030 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v036, v031, v033, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v001, 16, v001	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v030, 0xffff0000, v036	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_add_u32_e32 v000, 0x60000, v000	;.loc	4 100 196                       ; /tmp/tmp_67acr4h.cpp:100:196
	v_or_b32_sdwa v031, v030, v035 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_and_or_b32 v030, v034, s0, v001
	v_ashrrev_i32_e32 v001, 31, v000	;.loc	4 100 17                        ; /tmp/tmp_67acr4h.cpp:100:17
	v_lshl_add_u64 v000 v001, v000 v001, 1, s[8:9]
	global_store_dwordx2 v000 v001, v030 v031, off	;.loc	4 100 296 is_stmt 0             ; /tmp/tmp_67acr4h.cpp:100:296
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 296                         ; /tmp/tmp_67acr4h.cpp:0:296
	v_and_b32_e32 v030, 0x7f800000, v026	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v030	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v030, v026, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v030, v026, v030, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v030, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v031, 0x10000, v026	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v026, v030 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v030, v031, v026, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v026, 0x7f800000, v027	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v026	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v026, v027, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v031, v027, v026, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v026, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v031, 0x10000, v027
	v_cmp_eq_u32_sdwa vcc, v027, v026 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v031, v031, v027, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v026, 0x7f800000, v028	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v026	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v026, v028, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v032, v028, v026, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v026, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v027, 0x10000, v028
	v_cmp_eq_u32_sdwa vcc, v028, v026 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v032, v027, v028, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v026, 0x7f800000, v029	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v026	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v026, v029, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v033, v029, v026, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v026, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v027, 0x10000, v029
	v_cmp_eq_u32_sdwa vcc, v029, v026 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v033, v027, v029, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v026, 16, v030	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v027, 0xffff0000, v033	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v027, v027, v032 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v026, v031, s0, v026
	global_store_dwordx2 v000 v001, v026 v027, off offset:32	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 296 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:0:296
	v_and_b32_e32 v026, 0x7f800000, v022	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v026	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v026, v022, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v026, v022, v026, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v026, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v027, 0x10000, v022	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v022, v026 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v026, v027, v022, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v022, 0x7f800000, v023	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v022	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v022, v023, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v027, v023, v022, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v022, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v027, 0x10000, v023
	v_cmp_eq_u32_sdwa vcc, v023, v022 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v027, v027, v023, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v022, 0x7f800000, v024	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v022	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v022, v024, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v028, v024, v022, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v022, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v023, 0x10000, v024
	v_cmp_eq_u32_sdwa vcc, v024, v022 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v028, v023, v024, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v022, 0x7f800000, v025	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v022	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v022, v025, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v029, v025, v022, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v022, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v023, 0x10000, v025
	v_cmp_eq_u32_sdwa vcc, v025, v022 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v029, v023, v025, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v022, 16, v026	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v023, 0xffff0000, v029	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v023, v023, v028 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v022, v027, s0, v022
	global_store_dwordx2 v000 v001, v022 v023, off offset:64	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 296 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:0:296
	v_and_b32_e32 v022, 0x7f800000, v018	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v022	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v022, v018, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v022, v018, v022, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v022, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v023, 0x10000, v018	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v018, v022 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v022, v023, v018, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v018, 0x7f800000, v019	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v018	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v018, v019, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v023, v019, v018, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v018, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v023, 0x10000, v019
	v_cmp_eq_u32_sdwa vcc, v019, v018 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v023, v023, v019, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v018, 0x7f800000, v020	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v018	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v018, v020, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v024, v020, v018, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v018, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v019, 0x10000, v020
	v_cmp_eq_u32_sdwa vcc, v020, v018 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v024, v019, v020, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v018, 0x7f800000, v021	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v018	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v018, v021, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v025, v021, v018, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v018, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v019, 0x10000, v021
	v_cmp_eq_u32_sdwa vcc, v021, v018 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v025, v019, v021, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v018, 16, v022	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v019, 0xffff0000, v025	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v019, v019, v024 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v018, v023, s0, v018
	global_store_dwordx2 v000 v001, v018 v019, off offset:96	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 296 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:0:296
	v_and_b32_e32 v018, 0x7f800000, v014	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v018	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v018, v014, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v018, v014, v018, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v018, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v019, 0x10000, v014	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v014, v018 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v018, v019, v014, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v014, 0x7f800000, v015	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v014	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v014, v015, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v019, v015, v014, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v014, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v019, 0x10000, v015
	v_cmp_eq_u32_sdwa vcc, v015, v014 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v019, v019, v015, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v014, 0x7f800000, v016	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v014	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v014, v016, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v020, v016, v014, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v014, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v015, 0x10000, v016
	v_cmp_eq_u32_sdwa vcc, v016, v014 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v020, v015, v016, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v014, 0x7f800000, v017	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v014	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v014, v017, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v021, v017, v014, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v014, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v015, 0x10000, v017
	v_cmp_eq_u32_sdwa vcc, v017, v014 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v021, v015, v017, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v014, 16, v018	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v015, 0xffff0000, v021	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v015, v015, v020 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v014, v019, s0, v014
	global_store_dwordx2 v000 v001, v014 v015, off offset:128	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 296 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:0:296
	v_and_b32_e32 v014, 0x7f800000, v010	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v014	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v014, v010, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v014, v010, v014, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v014, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v015, 0x10000, v010	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v010, v014 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v014, v015, v010, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v010, 0x7f800000, v011	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v010	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v010, v011, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v015, v011, v010, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v010, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v015, 0x10000, v011
	v_cmp_eq_u32_sdwa vcc, v011, v010 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v015, v015, v011, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v010, 0x7f800000, v012	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v010	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v010, v012, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v016, v012, v010, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v010, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v011, 0x10000, v012
	v_cmp_eq_u32_sdwa vcc, v012, v010 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v016, v011, v012, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v010, 0x7f800000, v013	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v010	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v010, v013, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v017, v013, v010, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v010, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v011, 0x10000, v013
	v_cmp_eq_u32_sdwa vcc, v013, v010 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v017, v011, v013, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v010, 16, v014	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v011, 0xffff0000, v017	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v011, v011, v016 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v010, v015, s0, v010
	global_store_dwordx2 v000 v001, v010 v011, off offset:160	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 296 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:0:296
	v_and_b32_e32 v010, 0x7f800000, v006	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v010	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v010, v006, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v010, v006, v010, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v010, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v011, 0x10000, v006	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v006, v010 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v010, v011, v006, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v006, 0x7f800000, v007	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v006	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v006, v007, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v011, v007, v006, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v006, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v011, 0x10000, v007
	v_cmp_eq_u32_sdwa vcc, v007, v006 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v011, v011, v007, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v006, 0x7f800000, v008	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v006	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v006, v008, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v012, v008, v006, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v006, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v007, 0x10000, v008
	v_cmp_eq_u32_sdwa vcc, v008, v006 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v012, v007, v008, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v006, 0x7f800000, v009	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v006	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v006, v009, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v013, v009, v006, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v006, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v007, 0x10000, v009
	v_cmp_eq_u32_sdwa vcc, v009, v006 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v013, v007, v009, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v006, 16, v010	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v007, 0xffff0000, v013	;.loc	4 99 33                         ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v007, v007, v012 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v006, v011, s0, v006
	global_store_dwordx2 v000 v001, v006 v007, off offset:192	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_mov_b32 s0, 0x7f800000	;.loc	4 0 296 is_stmt 0               ; /tmp/tmp_67acr4h.cpp:0:296
	v_and_b32_e32 v006, 0x7f800000, v002	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v006	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v006, v002, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v006, v002, v006, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v006, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v007, 0x10000, v002	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_cmp_eq_u32_sdwa vcc, v002, v006 src0_sel:WORD_0 src1_sel:DWORD	;.loc	2 119 16                        ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	s_nop 1
	v_cndmask_b32_e32 v006, v007, v002, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v002, 0x7f800000, v003	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v002	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v002, v003, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v007, v003, v002, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v002, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v007, 0x10000, v003
	v_cmp_eq_u32_sdwa vcc, v003, v002 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v007, v007, v003, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v002, 0x7f800000, v004	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v002	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v002, v004, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v008, v004, v002, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v002, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v003, 0x10000, v004
	v_cmp_eq_u32_sdwa vcc, v004, v002 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v008, v003, v004, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v002, 0x7f800000, v005	;.loc	2 101 18 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:18
	v_cmp_ne_u32_e32 vcc, s0, v002	;.loc	2 101 9 is_stmt 0               ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:101:9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_bfe_u32 v002, v005, 16, 1	;.loc	2 118 44 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:44
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v009, v005, v002, s2	;.loc	2 118 15 is_stmt 0              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:118:15
	s_andn2_saveexec_b64 s[0:1], s[0:1]	;.loc	2 0 15                          ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:15
	v_mov_b32_e32 v002, 0	;.loc	2 119 16 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:119:16
	v_or_b32_e32 v003, 0x10000, v005
	v_cmp_eq_u32_sdwa vcc, v005, v002 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v009, v003, v005, vcc
	s_or_b64 exec, exec, s[0:1]	;.loc	2 0 16 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:16
	v_lshrrev_b32_e32 v002, 16, v006	;.loc	2 130 35 is_stmt 1              ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:130:35
	s_mov_b32 s0, 0xffff0000	;.loc	2 0 35 is_stmt 0                ; /opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bfloat16.h:0:35
	v_and_b32_e32 v003, 0xffff0000, v009	;.loc	4 99 33 is_stmt 1               ; /tmp/tmp_67acr4h.cpp:99:33
	v_or_b32_sdwa v003, v003, v008 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v002, v007, s0, v002
	global_store_dwordx2 v000 v001, v002 v003, off offset:224	;.loc	4 100 296                       ; /tmp/tmp_67acr4h.cpp:100:296
	s_endpgm	;.loc	4 102 1                         ; /tmp/tmp_67acr4h.cpp:102:1
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
...
	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
