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
def load_if(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.load(block_ptr)
    if EVEN_M:
        return tl.load(block_ptr, boundary_check=(1, ), padding_option="zero")
    if EVEN_N:
        return tl.load(block_ptr, boundary_check=(0, ), padding_option="zero")

    return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")


@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        tl.store(block_ptr, value)
        return
    if EVEN_M:
        tl.store(block_ptr, value, boundary_check=(1, ))
        return
    if EVEN_N:
        tl.store(block_ptr, value, boundary_check=(0, ))
        return
    tl.store(block_ptr, value, boundary_check=(0, 1))


@triton.jit
def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset, TYPE: tl.constexpr):
    tril_causal = q_offset[:, None] >= k_offset[None, :]
    triu_causal = q_offset[:, None] <= k_offset[None, :]

    if TYPE == 1:
        return ((triu_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) |
                (q_offset[:, None] == k_offset[None, :]))
    return ((tril_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) |
            (q_offset[:, None] == k_offset[None, :]))


def keep(config):
    m = config.kwargs["BLOCK_M"]
    n = config.kwargs["BLOCK_N"]
    return m % n == 0


@triton.autotune(list(filter(keep, fwd_autotune_config())), key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"])
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
    MASK_FN: tl.constexpr,
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
        if MASK_FN & 1:
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
        mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
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


def batched_attention(q, k, v, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale, mask_opt,
                      sparse_opt):
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
        MASK_FN=mask_opt,
        SPARSE_OPT=sparse_opt,
    )
    return o


def random_segments(
    total: int,
    num_segments: int,
    target_stddev: float,
    stddev_tol: float = 1.0,
    max_length: int | None = None,
    max_trials: int = 1000,
    device: str = "xpu",
    seed: int | None = None,
):
    """
    Generate random segment boundaries with a target segment length stddev.

    Returns:
        boundaries: (num_segments + 1,)
        lengths: (num_segments,)
        max_len: int
        stddev: float
    """
    if total <= 0:
        raise ValueError("total must be > 0")
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")
    if total < num_segments:
        raise ValueError("total must be >= num_segments to keep all segments non-empty")
    if target_stddev < 0:
        raise ValueError("target_stddev must be >= 0")

    if seed is not None:
        torch.manual_seed(seed)

    mean_len = total / num_segments
    target_cv = target_stddev / mean_len if mean_len > 0 else 0.0

    # Dirichlet concentration controls variability:
    # larger alpha -> lower variance
    alpha = max(0.01, 1.0 / (target_cv**2 + 1e-12))

    for _ in range(max_trials):
        probs = torch.distributions.Dirichlet(torch.full((num_segments, ), alpha)).sample()
        lengths = torch.round(probs * total).long()
        lengths = torch.clamp(lengths, min=1)

        diff = total - int(lengths.sum().item())

        while diff > 0:
            idx = torch.randint(num_segments, (1, ))
            lengths[idx] += 1
            diff -= 1

        while diff < 0:
            valid = torch.nonzero(lengths > 1).squeeze()
            if valid.numel() == 0:
                break
            ridx = torch.randint(valid.numel(), (1, ))
            idx = valid[ridx]
            lengths[idx] -= 1
            diff += 1

        stddev = lengths.float().std(unbiased=False)

        if max_length is not None and int(lengths.max().item()) > max_length:
            continue

        if abs(float(stddev.item()) - target_stddev) > stddev_tol:
            continue

        boundaries = torch.cat([torch.tensor([0], dtype=lengths.dtype), lengths.cumsum(0)]).to(device=device)

        return boundaries, lengths, int(lengths.max().item()), float(stddev.item())

    raise RuntimeError(f"Failed to sample segments with target_stddev={target_stddev} "
                       f"within tolerance={stddev_tol} after {max_trials} trials")


def segment_stats(boundaries: torch.Tensor):
    lengths = boundaries[1:] - boundaries[:-1]

    max_length = lengths.max().item()

    mean = lengths.float().mean()
    std = lengths.float().std(unbiased=False)  # population std
    cv = (std / mean).item()

    return {
        "lengths": lengths,
        "max_length": max_length,
        "cv": cv,
    }


def build_cases() -> list[dict[str, float | int | tuple[int, ...]]]:
    H_Q, H_KV, D_HEAD_QK, D_HEAD_V = 8, 8, 64, 64
    cases = [{
        "total_tokens": 289239, "num_segments": 256, "segment_stddev_over_mean": stddev, "H_Q": H_Q, "H_KV": H_KV,
        "D_HEAD_QK": D_HEAD_QK, "D_HEAD_V": D_HEAD_V
    } for stddev in [0.5, 5.0, 50.0, 500.0, 1000.0]]
    return cases


SEGMENT_CASES = build_cases()


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
        case["segment_stddev_over_mean"],
        case["H_Q"],
        case["H_KV"],
        case["D_HEAD_QK"],
        case["D_HEAD_V"],
    ] for case in SEGMENT_CASES]

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=[
                "TOTAL_TOKENS",
                "NUM_SEGMENTS",
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
    def benchmark(TOTAL_TOKENS, NUM_SEGMENTS, SEGMENT_STDDEV_OVER_MEAN, H_Q, H_KV, D_HEAD_QK, D_HEAD_V, provider):
        do_bench = benchmark_suite.get_do_bench(n_warmup=400, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        mean_len = TOTAL_TOKENS / NUM_SEGMENTS
        segments, _, max_len, _ = random_segments(TOTAL_TOKENS, NUM_SEGMENTS, SEGMENT_STDDEV_OVER_MEAN * mean_len,
                                                  seed=42)
        bitmap = build_segment_bitmap(segments, TOTAL_TOKENS, "xpu")
        dtype = torch.float16
        q = torch.randn((TOTAL_TOKENS, H_Q, D_HEAD_QK), dtype=dtype, device="xpu")
        k = torch.randn((TOTAL_TOKENS, H_KV, D_HEAD_QK), dtype=dtype, device="xpu")
        v = torch.randn((TOTAL_TOKENS, H_KV, D_HEAD_V), dtype=dtype, device="xpu")
        scale = 0.125

        if provider == "triton":
            triton_fn = lambda: batched_attention(q, k, v, bitmap, bitmap, segments, segments, max_len, scale, 1, False)
            _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        lengths = segments[1:] - segments[:-1]
        total_pairs = int(((lengths * (lengths + 1)) // 2).sum().item())
        tflops = lambda ms: (2 * total_pairs * H_Q * (D_HEAD_QK + D_HEAD_V) * 1e-12) / (ms * 1e-3)
        moved_bytes = ((q.numel() + k.numel() + v.numel()) * q.element_size() +
                       TOTAL_TOKENS * H_Q * D_HEAD_V * q.element_size())
        gbps = lambda ms: (moved_bytes * 1e-9) / (ms * 1e-3)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    get_benchmark().run(show_plots=False, print_data=True)
