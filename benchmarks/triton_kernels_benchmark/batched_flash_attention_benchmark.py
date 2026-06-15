import math
from typing import List, Optional

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite


def fwd_autotune_config() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=16),
    ]


@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.store(block_ptr, value)
    if EVEN_M:
        return tl.store(block_ptr, value, boundary_check=(1, ))
    if EVEN_N:
        return tl.store(block_ptr, value, boundary_check=(0, ))
    return tl.store(block_ptr, value, boundary_check=(0, 1))


@triton.jit
def segment_mask(q_attn_arg, k_attn_arg, q_offset, k_offset, MASK_TYPE: tl.constexpr):
    tril_causal = q_offset[:, None] >= k_offset[None, :]
    triu_causal = q_offset[:, None] <= k_offset[None, :]

    if MASK_TYPE == 1:
        return ((triu_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) |
                (q_offset[:, None] == k_offset[None, :]))
    return ((tril_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) |
            (q_offset[:, None] == k_offset[None, :]))


def keep(config: triton.Config) -> bool:
    block_m = config.kwargs["BLOCK_M"]
    block_n = config.kwargs["BLOCK_N"]
    return block_m % block_n == 0


@triton.autotune(list(filter(keep, fwd_autotune_config())), key=["QK_DIM", "V_DIM", "MASK_TYPE", "SPARSE_OPT"])
@triton.jit
def fa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    q_attn_arg_ptr,
    k_attn_arg_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    q_head,
    kv_head,
    scale,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    SPARSE_OPT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = o_ptr.type.element_ty
    start_m = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start

    if SPARSE_OPT:
        if k_len == 0:
            return
        begin = 0
        end = k_len
    else:
        if MASK_TYPE & 1:
            begin = start_m * BLOCK_M
            if begin >= k_len:
                return
            end = k_len
        else:
            begin = 0
            end = tl.minimum((start_m + 1) * BLOCK_M, k_len)

    log2e: tl.constexpr = 1.4426950408889634
    scale = scale.to(tl.float32)
    qk_scale = scale * log2e
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)

    q_base = q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM
    k_base = k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM
    v_base = v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM
    o_base = o_ptr + q_start * q_head * V_DIM + start_qh * V_DIM

    desc_q = tl.make_tensor_descriptor(
        q_base,
        shape=[q_len, QK_DIM],
        strides=[q_head * QK_DIM, 1],
        block_shape=[BLOCK_M, QK_DIM],
    )
    desc_k = tl.make_tensor_descriptor(
        k_base,
        shape=[k_len, QK_DIM],
        strides=[kv_head * QK_DIM, 1],
        block_shape=[BLOCK_N, QK_DIM],
    )
    desc_v = tl.make_tensor_descriptor(
        v_base,
        shape=[k_len, V_DIM],
        strides=[kv_head * V_DIM, 1],
        block_shape=[BLOCK_N, V_DIM],
    )
    desc_o = tl.make_tensor_descriptor(
        o_base,
        shape=[q_len, V_DIM],
        strides=[q_head * V_DIM, 1],
        block_shape=[BLOCK_M, V_DIM],
    )
    l_block_ptr = tl.make_block_ptr(base=l_ptr + q_start * q_head + start_qh, shape=(q_len, ), strides=(q_head, ),
                                    offsets=(start_m * BLOCK_M, ), block_shape=(BLOCK_M, ), order=(0, ))
    desc_q_attn_arg = tl.make_tensor_descriptor(
        q_attn_arg_ptr + q_start,
        shape=[q_len],
        strides=[1],
        block_shape=[BLOCK_M],
    )
    desc_k_attn_arg = tl.make_tensor_descriptor(
        k_attn_arg_ptr + k_start,
        shape=[k_len],
        strides=[1],
        block_shape=[BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, V_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M, ), value=-2**30, dtype=tl.float32)
    l = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    q = desc_q.load([start_m * BLOCK_M, 0])
    q_attn_arg = desc_q_attn_arg.load([start_m * BLOCK_M])

    for start_n in tl.range(begin, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N).to(tl.int32)
        k_attn_arg = desc_k_attn_arg.load([start_n])
        offset_n = start_n + tl.arange(0, BLOCK_N)
        mask = segment_mask(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_TYPE)
        if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
            k = desc_k.load([start_n, 0]).T
            s = tl.dot(q, k)
            boundary_mask = (offset_n < k_len)[None, :]
            s = tl.where(mask & boundary_mask, s, -2**30)
            m_new = tl.maximum(m, tl.max(s, 1))
            alpha = tl.math.exp2((m - m_new) * qk_scale)
            p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
            p_sum = tl.sum(p, 1)
            acc *= alpha[:, None]
            v = desc_v.load([start_n, 0])
            acc += tl.dot(p.to(dtype), v)
            l = l * alpha + p_sum
            m = m_new

    acc = acc / l[:, None]
    l = m * scale + tl.log(l)
    desc_o.store([start_m * BLOCK_M, 0], acc.to(dtype))
    store_if(l_block_ptr, l, False, True)


class FlashAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale,
                mask_type, sparse_opt):
        del ctx, max_seqlen_k
        q_len, q_head, qk_dim = q.shape
        _, kv_head, v_dim = v.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        o = q.new_empty(q_len, q_head, v_dim)
        l = q.new_empty(q_len, q_head, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), q_head, batch_size)
        fa_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            l,
            q_attn_arg,
            k_attn_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            q_head,
            kv_head,
            scale,
            QK_DIM=qk_dim,
            V_DIM=v_dim,
            MASK_TYPE=mask_type,
            SPARSE_OPT=sparse_opt,
        )
        return o

    @staticmethod
    def backward(ctx, do):
        del ctx, do
        raise NotImplementedError("Backward is not implemented for this benchmark kernel.")


SEGMENTS = (
    0,
    815,
    2662,
    3696,
    4205,
    4260,
    6400,
    7692,
    9724,
    11612,
    13456,
    15059,
    15063,
    16247,
    16472,
    18443,
    20446,
    22620,
    22737,
    23162,
    23975,
    25445,
    25742,
    26565,
    27804,
    30076,
    31338,
    33447,
    35157,
    35795,
    36803,
    38544,
    38644,
    39378,
    39881,
    41685,
    41822,
    42406,
    44670,
    44946,
    45490,
    47751,
    49294,
    50984,
    52108,
    52833,
    53098,
    54556,
    56467,
    58098,
    59197,
    60809,
    62970,
    63847,
    64881,
    65983,
    67028,
    67515,
    67706,
    70003,
    70423,
    72233,
    74482,
    74989,
    77367,
    79581,
    80151,
    80601,
    81491,
    82091,
    82726,
    84478,
    85556,
    87059,
    88251,
    88692,
    89212,
    90458,
    90882,
    92738,
    93401,
    95358,
    97675,
    98416,
    100023,
    101428,
    102701,
    104792,
    106530,
    106807,
    107291,
    108186,
    108337,
    108944,
    109878,
    111355,
    112949,
    114344,
    114561,
    116084,
    118253,
    118556,
    120632,
    121577,
    123437,
    123764,
    125412,
    125954,
    127380,
    127924,
    128987,
    129938,
    130291,
    132373,
    132847,
    133707,
    133717,
    134601,
    136841,
    137322,
    137864,
    139213,
    141481,
    142857,
    144252,
    144257,
    145789,
    147496,
    149144,
    151172,
    152389,
    152645,
    154002,
    154082,
    154914,
    156059,
    156725,
    157045,
    159061,
    159280,
    161568,
    163440,
    165193,
    165376,
    166290,
    168111,
    170436,
    170698,
    171371,
    173469,
    173537,
    174810,
    176698,
    178118,
    179830,
    180269,
    181195,
    181208,
    181950,
    182610,
    184040,
    184653,
    185667,
    186563,
    188779,
    190454,
    190526,
    191035,
    191048,
    192824,
    192861,
    193942,
    194282,
    195820,
    196073,
    196314,
    196441,
    197905,
    198502,
    200360,
    202601,
    204139,
    204981,
    205451,
    206656,
    207981,
    208951,
    209998,
    211277,
    211808,
    212506,
    214416,
    215648,
    216853,
    217601,
    219416,
    220891,
    223061,
    223235,
    224385,
    226595,
    227784,
    228919,
    231246,
    232673,
    233371,
    233950,
    234809,
    236214,
    238203,
    240320,
    242388,
    242616,
    244742,
    247017,
    247918,
    249400,
    251482,
    252192,
    252693,
    253478,
    255100,
    255659,
    256491,
    256725,
    257184,
    258575,
    260887,
    262431,
    263829,
    263851,
    264037,
    266019,
    266156,
    267827,
    268054,
    269866,
    270622,
    270757,
    270932,
    271567,
    271670,
    272503,
    273925,
    275065,
    276403,
    278555,
    278973,
    280973,
    281804,
    283180,
    284834,
    285496,
    287827,
    288868,
    289109,
    289239,
)

SEGMENT_PREFIXES = (33, 65, 129, len(SEGMENTS))


def segment_stats(boundaries: List[int]) -> dict[str, float | int]:
    lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean)**2 for length in lengths) / len(lengths)
    return {
        "num_segments": len(lengths),
        "total_tokens": boundaries[-1],
        "max_segment_length": max(lengths),
        "segment_stddev_over_mean": math.sqrt(variance) / mean,
    }


def build_cases() -> list[dict[str, float | int | tuple[int, ...]]]:
    cases = []
    for boundaries_count in SEGMENT_PREFIXES:
        boundaries = SEGMENTS[:boundaries_count]
        stats = segment_stats(list(boundaries))
        cases.append({
            **stats,
            "boundaries": boundaries,
        })
    return cases


SEGMENT_CASES = build_cases()


def find_segment_case(total_tokens: int, num_segments: int) -> dict[str, float | int | tuple[int, ...]]:
    for case in SEGMENT_CASES:
        if case["total_tokens"] == total_tokens and case["num_segments"] == num_segments:
            return case
    raise KeyError(f"No segment case for total_tokens={total_tokens}, num_segments={num_segments}")


def build_segment_bitmap(boundaries: torch.Tensor, total_tokens: int, device: str) -> torch.Tensor:
    bitmap = torch.zeros(total_tokens, device=device, dtype=torch.int64)
    bitmap[boundaries[:-1]] = 1
    return bitmap


def get_benchmark(providers_filter: Optional[List[str]] = None):
    supported_providers = {
        "triton": "Triton",
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    x_vals = [[
        case["total_tokens"],
        case["num_segments"],
        case["max_segment_length"],
        round(case["segment_stddev_over_mean"], 6),
        8,
        8,
        64,
        64,
    ] for case in SEGMENT_CASES]

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=[
                "TOTAL_TOKENS",
                "NUM_SEGMENTS",
                "MAX_SEGMENT_LENGTH",
                "SEGMENT_STDDEV_OVER_MEAN",
                "H_Q",
                "H_KV",
                "D_HEAD_QK",
                "D_HEAD_V",
            ],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=providers.keys(),
            line_names=providers.values(),
            styles=[("green", "-")],
            ylabel=["GB/s", "TFlops"],
            plot_name="batched-flash-attn-performance",
            args={},
        ))
    def benchmark(TOTAL_TOKENS, NUM_SEGMENTS, MAX_SEGMENT_LENGTH, SEGMENT_STDDEV_OVER_MEAN, H_Q, H_KV, D_HEAD_QK,
                  D_HEAD_V, provider):
        do_bench = benchmark_suite.get_do_bench(n_warmup=400, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        case = find_segment_case(TOTAL_TOKENS, NUM_SEGMENTS)
        assert case["max_segment_length"] == MAX_SEGMENT_LENGTH
        assert math.isclose(case["segment_stddev_over_mean"], SEGMENT_STDDEV_OVER_MEAN, rel_tol=0.0, abs_tol=1e-6)
        boundaries = torch.tensor(case["boundaries"], device="xpu", dtype=torch.int64)
        lengths = boundaries[1:] - boundaries[:-1]
        bitmap = build_segment_bitmap(boundaries, TOTAL_TOKENS, "xpu")
        dtype = torch.float16
        q = torch.randn((TOTAL_TOKENS, H_Q, D_HEAD_QK), dtype=dtype, device="xpu")
        k = torch.randn((TOTAL_TOKENS, H_KV, D_HEAD_QK), dtype=dtype, device="xpu")
        v = torch.randn((TOTAL_TOKENS, H_KV, D_HEAD_V), dtype=dtype, device="xpu")
        scale = 0.125

        if provider == "triton":
            triton_fn = lambda: FlashAttentionFunc.apply(q, k, v, bitmap, bitmap, boundaries, boundaries,
                                                         case["max_segment_length"], case["max_segment_length"], scale, 1,
                                                         False)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        total_pairs = sum(int(length.item()) * (int(length.item()) + 1) // 2 for length in lengths)
        tflops = lambda ms: (2 * total_pairs * H_Q * (D_HEAD_QK + D_HEAD_V) * 1e-12) / (ms * 1e-3)
        moved_bytes = ((q.numel() + k.numel() + v.numel()) * q.element_size() +
                       TOTAL_TOKENS * H_Q * D_HEAD_V * q.element_size())
        gbps = lambda ms: (moved_bytes * 1e-9) / (ms * 1e-3)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    get_benchmark().run(show_plots=False, print_data=True)
