import torch
import triton
import triton.language as tl

from torch.profiler import profile, ProfilerActivity


def fwd_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=16),
    ]


@triton.jit
def load_if(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.load(block_ptr)
    elif EVEN_M:
        return tl.load(block_ptr, boundary_check=(1, ), padding_option='zero')
    elif EVEN_N:
        return tl.load(block_ptr, boundary_check=(0, ), padding_option='zero')
    else:
        return tl.load(block_ptr, boundary_check=(0, 1), padding_option='zero')


@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.store(block_ptr, value)
    elif EVEN_M:
        return tl.store(block_ptr, value, boundary_check=(1, ))
    elif EVEN_N:
        return tl.store(block_ptr, value, boundary_check=(0, ))
    else:
        return tl.store(block_ptr, value, boundary_check=(0, 1))


@triton.jit
def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset, TYPE: tl.constexpr):
    tril_causal = q_offset[:, None] >= k_offset[None, :]
    triu_causal = q_offset[:, None] <= k_offset[None, :]

    if TYPE == 1:
        return ((triu_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) |
                (q_offset[:, None] == k_offset[None, :]))
    if TYPE == 2:
        return ((tril_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) |
                (q_offset[:, None] == k_offset[None, :]))


def keep(config):
    m = config.kwargs['BLOCK_M']
    n = config.kwargs['BLOCK_N']
    return m % n == 0


@triton.autotune(list(filter(keep, fwd_autotune_config())), key=['QK_DIM', 'V_DIM', 'MASK_FN', 'SPARSE_OPT'])
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
    DTYPE: tl.constexpr,
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
        begin = 0
        if k_len == 0:
            return
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
    # tl.store(l_block_ptr, l, boundary_check=(0, ))


class FlashAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale,
                mask_fn, sparse_opt):
        q_len, q_head, qk_dim = q.shape
        k_len, kv_head, v_dim = v.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        o = q.new_empty(q_len, q_head, v_dim)
        l = q.new_empty(q_len, q_head, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(max_seqlen_q, META['BLOCK_M']), q_head, batch_size)
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
            MASK_FN=mask_fn,
            SPARSE_OPT=sparse_opt,
            DTYPE=(19 if q.data == torch.float16 else 14),
        )
        ctx.save_for_backward(q, k, v, o, l, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scale = scale
        ctx.mask_fn = mask_fn
        ctx.sparse_opt = sparse_opt
        ctx.k_len = k_len
        ctx.q_head = q_head
        ctx.kv_head = kv_head
        ctx.qk_dim = qk_dim
        ctx.v_dim = v_dim
        ctx.batch_size = batch_size
        ctx.max_seqlen_k = max_seqlen_k
        ctx.dtype = (19 if q.data == torch.float16 else 14)
        return o

    @staticmethod
    def backward(ctx, do):
        pass


def random_segments(
    total: int,
    num_segments: int,
    target_cv: float = 0.5,
    cv_tol: float = 0.1,
    max_length: int | None = None,
    max_trials: int = 1000,
    device='xpu',
):
    '''
    Generate random segment boundaries.

    Returns:
        boundaries: (num_segments + 1,)
        lengths: (num_segments,)
        max_len: int
        cv: float
    '''

    # Dirichlet concentration controls variability:
    # larger alpha -> lower CV
    alpha = max(0.01, 1.0 / (target_cv**2))

    for _ in range(max_trials):
        # Sample positive proportions
        probs = torch.distributions.Dirichlet(torch.full((num_segments, ), alpha)).sample()

        # Convert to integer lengths
        lengths = torch.round(probs * total).long()

        # Ensure all segments are non-empty
        lengths = torch.clamp(lengths, min=1)

        # Fix rounding error to make sum exactly total
        diff = total - lengths.sum()

        while diff > 0:
            idx = torch.randint(num_segments, (1, ))
            lengths[idx] += 1
            diff -= 1

        while diff < 0:
            valid = torch.nonzero(lengths > 1).squeeze()
            if valid.numel() == 0:
                break
            idx = valid[torch.randint(len(valid), (1, ))]
            lengths[idx] -= 1
            diff += 1

        cv = lengths.float().std() / lengths.float().mean()

        if max_length is not None and lengths.max() > max_length:
            continue

        if abs(cv.item() - target_cv) > cv_tol:
            continue

        boundaries = torch.cat([torch.tensor([0]), lengths.cumsum(0)])

        return (boundaries.to(device=device), lengths, lengths.max().item(), cv.item())

    raise RuntimeError('Could not satisfy constraints. '
                       'Try increasing max_length, cv_tol, or max_trials.')


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


if __name__ == '__main__':

    use_user_segment = True
    if use_user_segment:
        segments = torch.tensor([
            0, 815, 2662, 3696, 4205, 4260, 6400, 7692, 9724, 11612, 13456, 15059, 15063, 16247, 16472, 18443, 20446,
            22620, 22737, 23162, 23975, 25445, 25742, 26565, 27804, 30076, 31338, 33447, 35157, 35795, 36803, 38544,
            38644, 39378, 39881, 41685, 41822, 42406, 44670, 44946, 45490, 47751, 49294, 50984, 52108, 52833, 53098,
            54556, 56467, 58098, 59197, 60809, 62970, 63847, 64881, 65983, 67028, 67515, 67706, 70003, 70423, 72233,
            74482, 74989, 77367, 79581, 80151, 80601, 81491, 82091, 82726, 84478, 85556, 87059, 88251, 88692, 89212,
            90458, 90882, 92738, 93401, 95358, 97675, 98416, 100023, 101428, 102701, 104792, 106530, 106807, 107291,
            108186, 108337, 108944, 109878, 111355, 112949, 114344, 114561, 116084, 118253, 118556, 120632, 121577,
            123437, 123764, 125412, 125954, 127380, 127924, 128987, 129938, 130291, 132373, 132847, 133707, 133717,
            134601, 136841, 137322, 137864, 139213, 141481, 142857, 144252, 144257, 145789, 147496, 149144, 151172,
            152389, 152645, 154002, 154082, 154914, 156059, 156725, 157045, 159061, 159280, 161568, 163440, 165193,
            165376, 166290, 168111, 170436, 170698, 171371, 173469, 173537, 174810, 176698, 178118, 179830, 180269,
            181195, 181208, 181950, 182610, 184040, 184653, 185667, 186563, 188779, 190454, 190526, 191035, 191048,
            192824, 192861, 193942, 194282, 195820, 196073, 196314, 196441, 197905, 198502, 200360, 202601, 204139,
            204981, 205451, 206656, 207981, 208951, 209998, 211277, 211808, 212506, 214416, 215648, 216853, 217601,
            219416, 220891, 223061, 223235, 224385, 226595, 227784, 228919, 231246, 232673, 233371, 233950, 234809,
            236214, 238203, 240320, 242388, 242616, 244742, 247017, 247918, 249400, 251482, 252192, 252693, 253478,
            255100, 255659, 256491, 256725, 257184, 258575, 260887, 262431, 263829, 263851, 264037, 266019, 266156,
            267827, 268054, 269866, 270622, 270757, 270932, 271567, 271670, 272503, 273925, 275065, 276403, 278555,
            278973, 280973, 281804, 283180, 284834, 285496, 287827, 288868, 289109, 289239
        ], device='xpu:0', dtype=torch.int64)

        lens, max_len, cv = segment_stats(segments)
    else:
        segments, lens, max_len, cv = random_segments(289239, 256)

    bitmap = torch.zeros(289239, device='xpu', dtype=torch.int64)
    bitmap[segments[:-1]] = 1

    dtype = torch.float16
    q = torch.randn([289239, 8, 64], dtype=dtype, device='xpu')
    k = torch.randn([289239, 8, 64], dtype=dtype, device='xpu')
    v = torch.randn([289239, 8, 64], dtype=dtype, device='xpu')
    # ctx, q, k, v, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, mask_fn, sparse_opt
    y = FlashAttentionFunc.apply(q, k, v, bitmap, bitmap, segments, segments, 2656, 2656, 0.125, 1, False)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(100):
            y = FlashAttentionFunc.apply(q, k, v, bitmap, bitmap, segments, segments, 2656, 2656, 0.125, 1, False)
    print(prof.key_averages().table(sort_by='self_xpu_time_total', row_limit=-1))
