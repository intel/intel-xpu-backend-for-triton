import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_grouped_kernel_final(
    # Input/Output Tensors
    Q,
    K_Buffer,
    V_Buffer,
    Req_to_tokens,
    B_Seqlen,
    Flat_Att_Out,
    # Metadata Tensors
    Task_metadata,
    # Parameters
    sm_scale,
    total_active_tasks,
    # Strides for tensor access
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs: tl.constexpr,
    stride_buf_kh: tl.constexpr,
    stride_buf_vbs: tl.constexpr,
    stride_buf_vh: tl.constexpr,
    # KV scales
    k_scale,
    v_scale,
    # Constexprs for kernel specialization
    kv_group_num: tl.constexpr,
    batch_size_q: tl.constexpr,
    batch_size_k: tl.constexpr,
    batch_size_v: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Stage 1 Kernel using static mapping. Each block processes one task.
    It reads task info from Task_metadata and writes its partial result
    to a unique slot in the flat Flat_Att_Out buffer.
    """
    task_id = tl.program_id(0)
    if task_id >= total_active_tasks:
        return

    # Fetch task details: (batch_idx, head_idx, split_idx, output_base_idx, num_splits)
    task_ptr = Task_metadata + task_id * 5
    cur_batch = tl.load(task_ptr + 0)
    cur_kv_head = tl.load(task_ptr + 1)
    cur_split = tl.load(task_ptr + 2)
    output_base_idx = tl.load(task_ptr + 3)
    num_splits = tl.load(task_ptr + 4)

    # cur_kv_head = cur_head // kv_group_num
    cur_head = cur_kv_head * kv_group_num + tl.arange(0, kv_group_num)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    split_start = cur_split * PAGE_SIZE
    split_end = tl.minimum(split_start + PAGE_SIZE, cur_batch_seq_len)

    if split_start >= cur_batch_seq_len:
        return

    # Core attention logic for the assigned split

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    # Scalar load
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_d[None, :], other=0.0)

    # 2d load
    # q_block_ptr = tl.make_block_ptr(
    #     base=Q,
    #     shape=(batch_size_q, BLOCK_DMODEL),
    #     strides=(stride_qh, 1),
    #     offsets=(cur_batch*(stride_qbs//stride_qh)+cur_kv_head*kv_group_num, 0),
    #     block_shape=(kv_group_num, BLOCK_DMODEL),
    #     order=(1,0)
    # )
    # q = tl.load(q_block_ptr, boundary_check=(0,1))

    # e_max = -float("inf")
    e_max = tl.full([kv_group_num, 1], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([kv_group_num, 1], dtype=tl.float32)
    acc = tl.zeros([kv_group_num, BLOCK_DV], dtype=tl.float32)

    dot_dtype = tl.float16
    k_scale_dot = k_scale.to(dot_dtype)
    v_scale_dot = v_scale.to(dot_dtype)

    k_head_offset = cur_kv_head * stride_buf_kh
    v_head_offset = cur_kv_head * stride_buf_vh

    for start_n in range(split_start, split_end, BLOCK_N):
        # (This inner loop logic is unchanged)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_end
        kv_page_number = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch + offs_n // PAGE_SIZE, mask=mask_n,
                                 other=0)

        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        # tl.static_print("kv_loc", kv_loc.shape)
        # Scalar load
        # offs_buf_k = (kv_loc[:, None] * stride_buf_kbs +
        #               k_head_offset + offs_d[None, :])
        # k = tl.load(K_Buffer + offs_buf_k,
        #     mask=(mask_n[:, None] & mask_d[None, :]), other=0)

        # 2d load
        # mask_n_scalar = cur_split*PAGE_SIZE<split_end
        kv_page_number_scalar = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch + cur_split)
        # tl.static_print("Used for block ptr", batch_size_k, stride_buf_kbs, kv_page_number_scalar, PAGE_SIZE, start_n, k_head_offset)
        # tl.device_print("Device kv page number", kv_page_number_scalar, task_id)
        k_block_ptr = tl.make_block_ptr(
            base=K_Buffer, shape=(batch_size_k, stride_buf_kbs), strides=(stride_buf_kbs, 1),
            offsets=(kv_page_number_scalar * PAGE_SIZE + start_n % PAGE_SIZE, k_head_offset),
            block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        # tl.static_print("k shape", k.shape)

        if kv_group_num < 8:
            q_r = q.reshape([kv_group_num, 1, BLOCK_DMODEL])
            qk = tl.sum(q_r * k.to(q_r.dtype) * k_scale, -1) * sm_scale
        else:
            k = tl.trans(k)
            qk = tl.dot(q.to(dot_dtype), k.to(dot_dtype) * k_scale_dot) * sm_scale  # Potential quantization

        qk = tl.where(mask_n, qk, float("-inf"))

        # Scalar load
        # offs_buf_v = (kv_loc[:, None] * stride_buf_vbs +
        #               v_head_offset + offs_dv[None, :])
        # v = tl.load(V_Buffer + offs_buf_v,
        #     mask=(mask_n[:, None]) & (mask_dv[None, :]), other=0)

        # 2d load
        v_block_ptr = tl.make_block_ptr(
            base=V_Buffer, shape=(batch_size_v, stride_buf_vbs), strides=(stride_buf_vbs, 1),
            offsets=(kv_page_number_scalar * PAGE_SIZE + start_n % PAGE_SIZE, v_head_offset),
            block_shape=(BLOCK_N, BLOCK_DV), order=(1, 0))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))

        # v = tl.trans(v)
        n_e_max = tl.maximum(tl.max(qk, 1, keep_dims=True), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc *= re_scale

        if kv_group_num < 8:
            p_r = p.reshape([kv_group_num, 1, BLOCK_N])
            v = tl.trans(v)
            acc += tl.sum(p_r.to(v.dtype) * v * v_scale, -1)
        else:
            acc += tl.dot(p.to(dot_dtype), v.to(dot_dtype) * v_scale_dot)

        e_sum = e_sum * re_scale + tl.sum(p, 1, keep_dims=True)
        e_max = n_e_max

    # Store result in the flat intermediate buffer using the unique task_id
    # Each task_id corresponds to kv_group_num q heads
    acc_norm = acc / e_sum
    e_max = e_max + tl.log(e_sum)
    # acc_norm = tl.reshape(acc_norm, [kv_group_num])
    e_max = tl.reshape(e_max, [kv_group_num])

    offs_h = output_base_idx + tl.arange(0, kv_group_num) * num_splits
    offs_mid_o = offs_h[:, None] * (Lv + 1) + offs_dv[None, :]
    offs_mid_o_1 = offs_h * (Lv + 1) + Lv
    tl.store(Flat_Att_Out + offs_mid_o, acc_norm, mask=mask_dv[None, :])
    tl.store(Flat_Att_Out + offs_mid_o_1, e_max)


dump_file = torch.load("decode_attention_input_step20165.pto", weights_only=False)
device = "xpu:0"
q = dump_file["query"]
batch_size = q.size(0)
b_seq_len = dump_file["context_lens"]
SPLIT_SIZE = 64
task_metadata_list = []
stage2_metadata_list = []
task_offset = 0

num_heads_kv = 4
num_heads_q = 32

q = q[:, :num_heads_q]

# New result
for i in range(batch_size):
    seq_len = b_seq_len[i].item()
    num_splits = (seq_len + SPLIT_SIZE - 1) // SPLIT_SIZE

    q_head_start_offsets_for_batch_item = []
    for j in range(num_heads_q):
        q_head_start_offsets_for_batch_item.append(task_offset)
        # For each (batch, head) pair, record its offset and split count
        stage2_metadata_list.append((task_offset, num_splits))
        task_offset += num_splits

    for j in range(num_heads_kv):
        first_q_head_in_group = j * (num_heads_q // num_heads_kv)
        output_base_for_group = q_head_start_offsets_for_batch_item[first_q_head_in_group]

        # Create the tasks for Stage 1
        for k in range(num_splits):
            task_metadata_list.append((i, j, k, output_base_for_group + k, num_splits))

total_active_tasks = len(task_metadata_list)

task_metadata_tensor = torch.tensor(task_metadata_list, dtype=torch.int32)
stage2_metadata_tensor = torch.tensor(stage2_metadata_list, dtype=torch.int32)
total_active_tasks = total_active_tasks

# print("Are close", torch.allclose(task_metadata_tensor, dump_file["task_metadata_tensor"]))

grid_stage1 = (total_active_tasks, )

# New
q = q.to(device)
k_buffer = dump_file["key_cache"].to(device)
v_buffer = dump_file["value_cache"].to(device)
req_to_token = dump_file["block_tables"].to(device)
b_seq_len = dump_file["context_lens"].to(device)
# flat_attn_logits = dump_file["flat_attn_logits"].to(device)
task_metadata_tensor = task_metadata_tensor.to(device)
sm_scale = dump_file["scale"]
total_active_tasks = total_active_tasks
kv_group_num = num_heads_q // num_heads_kv
# BLOCK_DMODEL = dump_file["BLOCK_DMODEL"]
# BLOCK_DV = dump_file["BLOCK_DV"]
BLOCK_DMODEL = 128
BLOCK_DV = 128
page_size = 64
logit_cap = 0.0
# Lk = dump_file["Lk"]
# Lv = dump_file["Lv"]
Lk = 128
Lv = 128

# For block ptr
batch_size_q = q.size(0) * q.size(1)
batch_size_k = k_buffer.size(0) * k_buffer.size(1)  # * k_buffer.size(2)
batch_size_v = v_buffer.size(0) * v_buffer.size(1)  # * v_buffer.size(2)

flat_attn_logits = torch.empty(total_active_tasks * kv_group_num, Lv + 1, dtype=torch.float32, device=q.device)

for BLOCK_N in (16, ):
    for num_warps in (1, ):
        for num_stages in (1, ):

            def fwd_triton():
                _fwd_grouped_kernel_final[grid_stage1](
                    q,
                    k_buffer,
                    v_buffer,
                    req_to_token,
                    b_seq_len,
                    flat_attn_logits,
                    task_metadata_tensor,
                    sm_scale,
                    total_active_tasks,
                    req_to_token.stride(0),
                    q.stride(0),
                    q.stride(1),
                    k_buffer.stride(-3),
                    k_buffer.stride(-2),
                    v_buffer.stride(-3),
                    v_buffer.stride(-2),
                    1.0,
                    1.0,
                    kv_group_num=kv_group_num,
                    batch_size_q=batch_size_q,
                    batch_size_k=batch_size_k,
                    batch_size_v=batch_size_v,
                    BLOCK_DMODEL=BLOCK_DMODEL,
                    BLOCK_DV=BLOCK_DV,
                    BLOCK_N=BLOCK_N,
                    PAGE_SIZE=page_size,
                    logit_cap=logit_cap,
                    Lk=Lk,
                    Lv=Lv,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    grf_mode='large',
                )

            # with profile(
            #     activities=[
            #         ProfilerActivity.XPU
            #     ],
            #     record_shapes=True,
            # ) as prof:
            torch.xpu.synchronize()
            ms, min_ms, max_ms = triton.testing.do_bench(fwd_triton, quantiles=[0.5, 0.2, 0.8], rep=400)
            torch.xpu.synchronize()

            # prof.export_chrome_trace(f"decode_block_{BLOCK_N}_num_warps_{num_warps}_num_stages_{num_stages}.json")
            print(f"Setup BLOCK_N {BLOCK_N} num_warps {num_warps} num_stages {num_stages}")
            print(f"tl   average {ms*1000:.2f}us min {min_ms*1000:.2f}us max {max_ms*1000:.2f}us")
