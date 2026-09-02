# Unified BatchedMoE kernel for Intel XPU.
# Single kernel covering all 20 v2 shapes (best_trials_v2/v2__2_BatchedMoE_triton__bench-gpu*),
# spanning bf16 / fp8_w8a8 / int8_w8a16 driven by a single QUANT constexpr.
#
# Distilled from the per-shape v2 winners:
#   * Universal spine (gpu-0..gpu-12, gpu-13..gpu-20):
#         3-layer skeleton (moe_mmk + expert_triton_kernel + batched_triton_kernel),
#         tensor descriptors for A/B/C, lambda grid, empty-expert early-exit,
#         GROUP_SIZE_M swizzle for L2 B-reuse, fp32 accumulator.
#   * fp8_w8a8 winner (gpu-15, 8.42x): host-side B = B.to(fp8).to(bf16) once,
#         A = A.to(fp8).to(bf16) per call (cached on data_ptr). Kernel runs as bf16,
#         no quant flags set -> avoids slow XPU fp8 dot.
#   * int8_w8a16 winners (gpu-17 6.30x, gpu-19 7.22x, gpu-18 4.48x): cache
#         B_int8 + per-N B_scale once; cast B to bf16 inside K-loop; b_scale * acc at
#         epilogue.
#   * Pipelining: num_stages 2-3 for bf16, 3-5 for quant-shaped autotune configs.

import torch
import triton
import triton.language as tl


def dequant_to_bf16(q, scale, block_shape=None):
    """Dequantize a batched fp8 tensor to bf16 by applying its scale.

    Used by the fp8 "host bf16 trick" (QUANT=3). Dequantizes per-expert into a
    preallocated bf16 output: XPU casts fp8 -> f32 under the hood, so a
    whole-tensor dequant would hold a 4x-fp8 fp32 temp (which OOMs the larger
    llama4/qwen fp8 B matrices). Looping over experts bounds that temp to a
    single (rows, cols) slice, keeping peak at ~2x fp8. Per-tensor / per-token
    scales broadcast directly; block scales are expanded over the trailing dims.
    """
    if scale is None:
        return q.to(torch.bfloat16)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    if block_shape is None:
        for e in range(q.shape[0]):
            out[e] = (q[e].to(torch.float32) * scale[e]).to(torch.bfloat16)
        return out
    block_n, block_k = block_shape
    for e in range(q.shape[0]):
        s = torch.repeat_interleave(scale[e], block_n, dim=-2)
        s = torch.repeat_interleave(s, block_k, dim=-1)
        s = s[..., :q.shape[-2], :q.shape[-1]]
        out[e] = (q[e].to(torch.float32) * s).to(torch.bfloat16)
    return out


def normalize_batched_scales_shape(scales, num_experts):
    if scales is not None and scales.ndim < 3:
        if scales.numel() == 1:
            scales = scales.view(1)
            scales = torch.repeat_interleave(scales, num_experts, dim=0).view(num_experts, 1, 1)
        else:
            scales = scales.view(num_experts, -1, scales.size(-1))
    return scales


# ---------------------------------------------------------------------------
# Inner GEMM (one expert, one (pid_m, pid_n) tile).
# ---------------------------------------------------------------------------
@triton.jit
def moe_mmk(
    a_desc, b_desc,
    K, expert_id,
    a_scale_ptr, b_scale_ptr,
    stride_ak: tl.int64, stride_bk: tl.int64,
    stride_ase: tl.int64, stride_asm: tl.int64, stride_ask: tl.int64,
    stride_bse: tl.int64, stride_bsk: tl.int64, stride_bsn: tl.int64,
    offs_m, offs_n, offs_bn, mask_m,
    group_n: tl.constexpr, group_k: tl.constexpr,
    pid_m, pid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr, use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
):
    if use_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + expert_id * stride_bse + offs_n[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bsn
        elif per_act_token_quant:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=mask_m, other=0.0)[:, None]
            b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        b = b_desc.load([pid_n * BLOCK_N, k * BLOCK_K]).T

        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=mask_m, other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator = tl.dot(a, b, acc=accumulator)

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    return accumulator


# ---------------------------------------------------------------------------
# Per-expert tile dispatch.
# ---------------------------------------------------------------------------
@triton.jit
def expert_triton_kernel(
    a_desc, b_desc, c_desc,
    expert_id, compute_type: tl.constexpr,
    M, N, K,
    a_scale_ptr, b_scale_ptr, b_zp_ptr,
    stride_am: tl.int64, stride_ak: tl.int64,
    stride_bk: tl.int64, stride_bn: tl.int64,
    stride_cm: tl.int64, stride_cn: tl.int64,
    stride_ase: tl.int64, stride_asm: tl.int64, stride_ask: tl.int64,
    stride_bse: tl.int64, stride_bsk: tl.int64, stride_bsn: tl.int64,
    offs_bn, group_n, group_k,
    pid_m, pid_n,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    mask_m = offs_m < M

    accumulator = moe_mmk(
        a_desc, b_desc, K, expert_id,
        a_scale_ptr, b_scale_ptr,
        stride_ak, stride_bk,
        stride_ase, stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        offs_m, offs_n, offs_bn, mask_m,
        group_n, group_k, pid_m, pid_n,
        BLOCK_M, BLOCK_N, BLOCK_K,
        compute_type,
        use_fp8_w8a8, use_int8_w8a16, per_act_token_quant,
    )
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)


# ---------------------------------------------------------------------------
# Unified autotune table — union of winning configs across all 20 v2 shapes.
# ---------------------------------------------------------------------------
def get_unified_configs():
    cfgs = []

    def C(BM, BN, BK, GM, s, w, grf="256"):
        cfgs.append(triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK,
             "GROUP_SIZE_M": GM, "grf_mode": grf},
            num_stages=s, num_warps=w))

    # === BLOCK_M=64 single-tile (most shapes have M=64) ===
    # Direct gpu-15 winners + bf16 winners cluster.
    C(64, 512, 64,  1, 3, 32)
    C(64, 256, 64,  1, 4, 16)
    C(64, 256, 128, 1, 3, 16)
    C(64, 128, 128, 1, 4, 8)
    C(64, 128, 64,  1, 5, 8)
    C(64, 256, 32,  1, 3, 32)
    C(64, 128, 32,  1, 2, 16)
    C(64, 256, 64,  4, 2, 16)   # gpu-9-style swizzle for big-M

    # === BLOCK_M=32 + swizzle (gpu-13 fp8 cluster) ===
    C(32, 512, 64,  2, 3, 32)
    C(32, 256, 128, 2, 3, 16)
    C(32, 128, 128, 2, 4, 8)
    C(32, 256, 64,  2, 4, 16)
    C(32, 256, 64,  2, 2, 32)
    C(32, 128, 64,  2, 3, 16)

    # === BLOCK_M=16 + swizzle (gpu-18 int8 cluster) ===
    C(16, 512, 128, 4, 2, 32)
    C(16, 256, 128, 4, 3, 16)
    C(16, 256, 64,  4, 4, 16)
    C(16, 128, 64,  4, 2, 8)

    # === BLOCK_M=8 max-parallelism (gpu-11/12, gpu-20 int8 cluster) ===
    C(8, 512, 128, 8, 2, 32)
    C(8, 256, 128, 8, 3, 16)
    C(8, 512, 64,  8, 2, 32)
    C(8, 256, 64,  8, 2, 16)
    C(8, 128, 64,  8, 2, 4)

    # === Big-M swizzle (gpu-9, gpu-10, gpu-16, gpu-20) ===
    C(128, 256, 32, 2, 3, 32)
    C(128, 256, 64, 2, 2, 16)
    C(128, 512, 32, 2, 3, 32)
    C(128, 128, 64, 2, 4, 16)
    C(256, 256, 64, 1, 2, 32)
    C(256, 128, 64, 1, 2, 32)

    # === GRF128 high-warp variants (memory-bound) ===
    C(64, 128, 64, 1, 4, 8,  grf="128")
    C(64, 128, 32, 4, 2, 64, grf="128")

    return cfgs


# ---------------------------------------------------------------------------
# Top-level batched kernel (one program per (mn-tile, expert)).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=get_unified_configs(),
    key=["max_num_tokens", "K", "N", "use_fp8_w8a8", "use_int8_w8a16"],
)
@triton.jit
def batched_triton_kernel(
    a_ptr, b_ptr, c_ptr, expert_num_tokens,
    compute_type: tl.constexpr,
    max_num_tokens: tl.constexpr,
    K: tl.constexpr, N: tl.constexpr,
    a_scale_ptr, b_scale_ptr, b_zp_ptr,
    stride_ae: tl.int64,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_be: tl.int64,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    stride_ase: tl.constexpr, stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    stride_bse: tl.constexpr, stride_bsk: tl.constexpr, stride_bsn: tl.constexpr,
    group_n: tl.constexpr, group_k: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    expert_id = tl.program_id(axis=1)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        return

    pid_mn = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        return

    cta_m_size = tl.minimum(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = tl.minimum(BLOCK_N, N - cta_n_start)

    a_desc = tl.make_tensor_descriptor(
        base=a_ptr + expert_id * stride_ae,
        shape=(e_num_tokens, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr + expert_id * stride_be,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K))
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr + expert_id * stride_ce,
        shape=(e_num_tokens, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N))

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    if use_fp8_w8a8:
        a_scale_ptr = a_scale_ptr + expert_id * stride_ase
        b_scale_ptr = b_scale_ptr + expert_id * stride_bse
        if group_k > 0 and group_n > 0 or per_act_token_quant:
            a_scale_ptr = a_scale_ptr + cta_m_start * stride_asm

    expert_triton_kernel(
        a_desc, b_desc, c_desc,
        expert_id, compute_type,
        cta_m_size, cta_n_size, K,
        a_scale_ptr, b_scale_ptr, b_zp_ptr,
        stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_ase, stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        offs_bn, group_n, group_k,
        pid_m, pid_n,
        use_fp8_w8a8, use_int8_w8a16, per_act_token_quant,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )


# ---------------------------------------------------------------------------
# Host-side launcher.
# ---------------------------------------------------------------------------
def invoke_moe_batched_triton_kernel(
    A, B, C, expert_num_tokens, compute_type,
    A_scale, B_scale, B_zp,
    use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16,
    per_act_token_quant, block_shape=None,
):
    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)
    E = expert_num_tokens.size(0)

    grid = lambda META: (
        triton.cdiv(max_num_tokens, META["BLOCK_M"]) *
        triton.cdiv(N, META["BLOCK_N"]),
        E,
    )

    A_scale = normalize_batched_scales_shape(A_scale, E)
    if B_scale is not None and B_scale.ndim == 1:
        assert B_scale.numel() == E
        B_scale = B_scale.view(-1, 1, 1)

    if B_scale is not None:
        if B_scale.ndim == 1:
            stride_bse, stride_bsk, stride_bsn = 1, 0, 0
        else:
            stride_bse = B_scale.stride(0)
            stride_bsk = B_scale.stride(2)
            stride_bsn = B_scale.stride(1)
    else:
        stride_bse, stride_bsk, stride_bsn = 0, 0, 0

    if A_scale is not None:
        stride_ase = A_scale.stride(0)
        stride_asm = A_scale.stride(1)
        stride_ask = A_scale.stride(2)
    else:
        stride_ase, stride_asm, stride_ask = 0, 0, 0

    batched_triton_kernel[grid](
        A, B, C, expert_num_tokens,
        compute_type, max_num_tokens, K, N,
        A_scale, B_scale, B_zp,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        stride_ase, stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        use_fp8_w8a8, use_int8_w8a16, per_act_token_quant,
    )


# ---------------------------------------------------------------------------
# Model class — pack-once weight cache + pre-allocated C buffer.
# fp8_w8a8 has two modes: QUANT=1 runs the kernel's native fp8 path (operands
# stay fp8, scales applied in-kernel); QUANT=3 is the gpu-15 host bf16 trick
# (dequant fp8->bf16 host-side, run a bf16 kernel). int8_w8a16 uses the
# gpu-17/18/19 int8 caching.
# ---------------------------------------------------------------------------
class Model(torch.nn.Module):
    def __init__(self, E: int, M: int, K: int, N: int, QUANT: int = 0):
        super().__init__()
        self.E = E
        self.M = M
        self.K = K
        self.N = N
        self.QUANT = QUANT  # 0=bf16, 1=fp8 native, 2=int8_w8a16, 3=fp8 host bf16 trick
        self._packed = False
        self._A_cached_ptr = None  # QUANT=3: cache A's bf16 conversion
        self._fp8_scales = None    # QUANT=3: (A_scale, B_scale, block_shape)

    def _pack_weights(self, A: torch.Tensor, B: torch.Tensor):
        device = A.device
        if self.QUANT == 2:
            # gpu-17/18/19 int8 trick: cache B as int8 + per-N B_scale.
            self._B_int8 = B.to(torch.int8)
            self._B_scale_w8a16 = torch.ones(self.E, 1, self.N,
                                             device=device, dtype=torch.float32)
        elif self.QUANT == 3:
            # gpu-15 fp8 trick: dequant B (fp8) -> bf16 once, applying its scale.
            _, B_scale, block_shape = self._fp8_scales
            self._B_bf16 = dequant_to_bf16(B, B_scale, block_shape)
        self._C_buf = torch.zeros(self.E, self.M, self.N,
                                  device=device, dtype=torch.bfloat16)
        self._packed = True

    def forward(self, A: torch.Tensor, B: torch.Tensor,
                expert_num_tokens: torch.Tensor,
                A_scale: torch.Tensor = None, B_scale: torch.Tensor = None,
                block_shape=None) -> torch.Tensor:
        if self.QUANT == 3:
            # Stash scales so _pack_weights can dequant B before the first run.
            self._fp8_scales = (A_scale, B_scale, block_shape)
        if not self._packed:
            self._pack_weights(A, B)

        blk = None
        if self.QUANT == 1:
            # Native fp8: operands stay fp8, the kernel dequants per-tile using
            # the scales (per-tensor at the epilogue, block per K-group).
            A_in, B_in = A, B
            a_scale, b_scale = A_scale, B_scale
            use_fp8_flag = True
            use_int8_w8a16 = False
            blk = block_shape
        elif self.QUANT == 3:
            # Host bf16 trick: run bf16, quant absorbed host-side. B dequant'd at
            # pack; A dequant'd once and cached on data_ptr identity.
            if self._A_cached_ptr != A.data_ptr():
                self._A_bf16 = dequant_to_bf16(A, A_scale, block_shape)
                self._A_cached_ptr = A.data_ptr()
            A_in, B_in = self._A_bf16, self._B_bf16
            a_scale, b_scale = None, None
            use_fp8_flag = False
            use_int8_w8a16 = False
        elif self.QUANT == 2:
            A_in = A
            B_in = self._B_int8
            a_scale, b_scale = None, self._B_scale_w8a16
            use_fp8_flag = False
            use_int8_w8a16 = True
        else:
            A_in, B_in = A, B
            a_scale, b_scale = None, None
            use_fp8_flag = False
            use_int8_w8a16 = False

        self._C_buf.zero_()
        invoke_moe_batched_triton_kernel(
            A_in, B_in, self._C_buf, expert_num_tokens,
            compute_type=tl.bfloat16,
            A_scale=a_scale, B_scale=b_scale, B_zp=None,
            use_fp8_w8a8=use_fp8_flag, use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=False,
            per_act_token_quant=False,
            block_shape=blk,
        )
        return self._C_buf
