import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
from tilelang.profiler import do_bench
import math


@tilelang.jit(
    out_idx=[8],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def mla_decode_tilelang(
    batch, h_q__128, h_kv__1, max_seqlen_pad, dv__512, dpe, block_N__64, block_H__64, num_split, block_size, softmax_scale=None
):
    if softmax_scale is None:
        softmax_scale = (dv__512 + dpe) ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = h_q__128 // h_kv__1
    VALID_BLOCK_H__64 = min(block_H__64, kv_group_num)
    assert h_kv__1 == 1, "h_kv must be 1"
    assert block_size >= block_N__64 and block_size % block_N__64 == 0, "block_size must be larger than block_N and a multiple of block_N"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, h_q__128, dv__512], dtype),
        Q_pe: T.Tensor([batch, h_q__128, dpe], dtype),
        KV: T.Tensor([batch * max_seqlen_pad, h_kv__1, dv__512], dtype),
        K_pe: T.Tensor([batch * max_seqlen_pad, h_kv__1, dpe], dtype),
        block_table: T.Tensor([batch, max_seqlen_pad // block_size], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, h_q__128, num_split], dtype),
        Output_partial: T.Tensor([batch, h_q__128, num_split, dv__512], dtype),
        Output: T.Tensor([batch, h_q__128, dv__512], dtype),
    ):
        # split kv
        with T.Kernel(batch, h_q__128 // min(block_H__64, kv_group_num), num_split, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H__64, dv__512], dtype)
            S_shared = T.alloc_shared([block_H__64, block_N__64], dtype)
            Q_pe_shared = T.alloc_shared([block_H__64, dpe], dtype)
            KV_shared = T.alloc_shared([block_N__64, dv__512], dtype)
            K_pe_shared = T.alloc_shared([block_N__64, dpe], dtype)
            O_shared = T.alloc_shared([block_H__64, dv__512], dtype)
            acc_s = T.alloc_fragment([block_H__64, block_N__64], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H__64, block_N__64], dtype)
            acc_o = T.alloc_fragment([block_H__64, dv__512], accum_dtype)
            scores_max = T.alloc_fragment([block_H__64], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H__64], accum_dtype)
            scores_scale = T.alloc_fragment([block_H__64], accum_dtype)
            scores_sum = T.alloc_fragment([block_H__64], accum_dtype)
            logsum = T.alloc_fragment([block_H__64], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H__64)
            T.use_swizzle(10)

            T.copy(Q[bx, by * VALID_BLOCK_H__64 : (by + 1) * VALID_BLOCK_H__64, :], Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H__64 : (by + 1) * VALID_BLOCK_H__64, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            total_blocks = T.ceildiv(cache_seqlens[bx], block_N__64)
            blocks_per_split = T.floordiv(total_blocks, num_split)
            remaining_blocks = T.floormod(total_blocks, num_split)
            loop_range = blocks_per_split + T.if_then_else(bz < remaining_blocks, 1, 0)
            start = (blocks_per_split * bz + T.min(bz, remaining_blocks)) * block_N__64

            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = block_table[bx, (start + k * block_N__64) // block_size] * block_size + (k * block_N__64) % block_size
                T.copy(KV[kv_start : kv_start + block_N__64, cur_kv_head, :], KV_shared)
                T.copy(K_pe[kv_start : kv_start + block_N__64, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                for i, j in T.Parallel(block_H__64, block_N__64):
                    acc_s[i, j] = T.if_then_else(start + k * block_N__64 + j >= cache_seqlens[bx], -T.infinity(accum_dtype), acc_s[i, j])
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H__64):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H__64):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H__64, block_N__64):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                for i in T.Parallel(block_H__64):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H__64, dv__512):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H__64, dv__512):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H__64):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bx, by * VALID_BLOCK_H__64 : (by + 1) * VALID_BLOCK_H__64, bz])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bx, by * VALID_BLOCK_H__64 : (by + 1) * VALID_BLOCK_H__64, bz, :])

        # combine
        with T.Kernel(h_q__128, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dv__512], dtype)
            o_accum_local = T.alloc_fragment([dv__512], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bz, by, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bz, by, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            for k in T.serial(num_split):
                for i in T.Parallel(dv__512):
                    po_local[i] = Output_partial[bz, by, k, i]
                lse_local_split = glse[bz, by, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dv__512):
                    o_accum_local[i] += po_local[i] * scale_local
            for i in T.Parallel(dv__512):
                Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, h_q__128, dv__512], dtype),
        Q_pe: T.Tensor([batch, h_q__128, dpe], dtype),
        KV: T.Tensor([batch * max_seqlen_pad, h_kv__1, dv__512], dtype),
        K_pe: T.Tensor([batch * max_seqlen_pad, h_kv__1, dpe], dtype),
        block_table: T.Tensor([batch, max_seqlen_pad // block_size], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, h_q__128, num_split], dtype),
        Output_partial: T.Tensor([batch, h_q__128, num_split, dv__512], dtype),
        Output: T.Tensor([batch, h_q__128, dv__512], dtype),
    ):
        with T.Kernel(batch, h_q__128 // min(block_H__64, kv_group_num), threads=256) as (bx, by):
            Q_shared = T.alloc_shared([block_H__64, dv__512], dtype)
            S_shared = T.alloc_shared([block_H__64, block_N__64], dtype)
            Q_pe_shared = T.alloc_shared([block_H__64, dpe], dtype)
            KV_shared = T.alloc_shared([block_N__64, dv__512], dtype)
            K_pe_shared = T.alloc_shared([block_N__64, dpe], dtype)
            O_shared = T.alloc_shared([block_H__64, dv__512], dtype)
            acc_s = T.alloc_fragment([block_H__64, block_N__64], accum_dtype)
            acc_o = T.alloc_fragment([block_H__64, dv__512], accum_dtype)
            max_vec = T.alloc_fragment([block_H__64], accum_dtype)
            max_vec_prev = T.alloc_fragment([block_H__64], accum_dtype)
            scores_vec = T.alloc_fragment([block_H__64], accum_dtype)
            scores_sum = T.alloc_fragment([block_H__64], accum_dtype)
            logsum = T.alloc_fragment([block_H__64], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H__64)
            T.use_swizzle(10)

            T.copy(Q[bx, by * VALID_BLOCK_H__64 : (by + 1) * VALID_BLOCK_H__64, :], Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H__64 : (by + 1) * VALID_BLOCK_H__64, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(max_vec, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(cache_seqlens[bx], block_N__64)
            for kr in T.Pipelined(loop_range, num_stages=2):
                k = loop_range - 1 - kr
                kv_start = block_table[bx, (k * block_N__64) // block_size] * block_size + (k * block_N__64) % block_size
                T.copy(KV[kv_start : kv_start + block_N__64, cur_kv_head, :], KV_shared)
                T.copy(K_pe[kv_start : kv_start + block_N__64, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(max_vec, max_vec_prev)
                T.fill(max_vec, -T.infinity(accum_dtype))
                if kr == 0:
                    for i, j in T.Parallel(block_H__64, block_N__64):
                        acc_s[i, j] = T.if_then_else(k * block_N__64 + j >= cache_seqlens[bx], -T.infinity(accum_dtype), acc_s[i, j])
                T.reduce_max(acc_s, max_vec, dim=1, clear=False)
                for i in T.Parallel(block_H__64):
                    max_vec[i] = T.max(max_vec[i], max_vec_prev[i])
                for i in T.Parallel(block_H__64):
                    scores_vec[i] = T.exp2(max_vec_prev[i] * scale - max_vec[i] * scale)
                for i, j in T.Parallel(block_H__64, block_N__64):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - max_vec[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H__64):
                    logsum[i] = logsum[i] * scores_vec[i] + scores_sum[i]
                for i, j in T.Parallel(block_H__64, dv__512):
                    acc_o[i, j] *= scores_vec[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            for i, j in T.Parallel(block_H__64, dv__512):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bx, by * VALID_BLOCK_H__64 : (by + 1) * VALID_BLOCK_H__64, :])

    if num_split > 1:
        return main_split
    else:
        return main_no_split


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(
    q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q__1, cache_seqlens, h_q__128, h_kv__1, d__576, dv__512, causal, dtype
):
    # q: [b, s_q, h_q, d]
    # block_table: [b, max_seqlen_pad // block_size]
    # blocked_k: [b * max_seqlen_pad // block_size, block_size, h_kv, d]
    # cache_seqlens: [b]
    blocked_v = blocked_k[..., :dv__512]

    def ref_mla():
        out = torch.empty(b, s_q__1, h_q__128, dv__512, dtype=torch.float32, device=q.device)
        lse = torch.empty(b, h_q__128, s_q__1, dtype=torch.float32, device=q.device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv__1, d__576)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv__1, dv__512)[begin:end].transpose(0, 1),
                h_q__128,
                h_kv__1,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out.to(dtype), lse.to(dtype)

    out_torch, _ = ref_mla()
    return out_torch


def run_tilelang_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b___may_8192,
    s_q__1,
    cache_seqlens,
    h_q__128,
    h_kv__1,
    d__576,
    dv__512,
    causal,
    dtype,
):
    assert d__576 > dv__512, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv__512].contiguous(), q[..., dv__512:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv__512].contiguous(), blocked_k[..., dv__512:].contiguous()

    dpe = d__576 - dv__512
    num_kv_splits__1 = 1
    BLOCK_N = 64
    BLOCK_H = min(64, h_q__128 // h_kv__1)
    softmax_scale = d__576**-0.5

    out_partial = torch.empty(b___may_8192, h_q__128, num_kv_splits__1, dv__512, dtype=dtype, device=q.device)
    glse = torch.empty(b___may_8192, h_q__128, num_kv_splits__1, dtype=dtype, device=q.device)
    kernel = mla_decode_tilelang(
        b___may_8192, h_q__128, h_kv__1, max_seqlen_pad, dv__512, dpe, BLOCK_N, BLOCK_H, num_kv_splits__1, block_size, softmax_scale
    )
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    def flash_mla_tilelang():
        out = profiler.func(
            q_nope.view(-1, h_q__128, dv__512),
            q_pe.view(-1, h_q__128, dpe),
            blocked_k_nope.view(-1, h_kv__1, dv__512),
            blocked_k_pe.view(-1, h_kv__1, dpe),
            block_table,
            cache_seqlens,
            glse,
            out_partial,
        )
        return out.view([b___may_8192, s_q__1, h_q__128, dv__512])

    out_flash = flash_mla_tilelang()
    t = do_bench(flash_mla_tilelang)
    out_ref = run_torch_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        b___may_8192,
        s_q__1,
        cache_seqlens,
        h_q__128,
        h_kv__1,
        d__576,
        dv__512,
        causal,
        dtype,
    )
    torch.testing.assert_close(out_flash, out_ref, rtol=0.01, atol=0.01)
    print("All close")
    return out_flash, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DeepSeek-V2/V3 MLA 参数 (absorb 形式):
    # - kv_lora_rank = 512, qk_rope_head_dim = 64
    # - d = kv_lora_rank + qk_rope_head_dim = 576
    # - dv = kv_lora_rank = 512
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--h_q", type=int, default=128, help="q heads number (num_attention_heads)")
    parser.add_argument("--h_kv", type=int, default=1, help="kv heads number (1 after absorb)")
    parser.add_argument("--cache_seqlen", type=int, default=8192, help="kv cache context length")
    parser.add_argument("--d", type=int, default=576, help="query/key head dim = kv_lora_rank(512) + qk_rope_head_dim(64)")
    parser.add_argument("--dv", type=int, default=512, help="value head dim = kv_lora_rank")
    args = parser.parse_args()
    b, h_q__128, h_kv__1, cache_seqlen, d__576, dv__512 = args.batch, args.h_q, args.h_kv, args.cache_seqlen, args.d, args.dv

    device = "cuda"
    dtype = torch.float16

    s_q__1 = 1  # for decode, s_q = 1
    block_size = 64
    cache_seqlens = torch.tensor([cache_seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device)
    dpe__64 = d__576 - dv__512
    causal = True

    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256

    total_flops = s_q__1 * total_seqlens * h_q__128 * d__576 * 2

    q = torch.randn(b, s_q__1, h_q__128, d__576, dtype=dtype, device=device)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device=device).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv__1, d__576, dtype=dtype, device=device)
    out_flash, latency = run_tilelang_mla(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q__1, cache_seqlens, h_q__128, h_kv__1, d__576, dv__512, causal, dtype
    )

    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
