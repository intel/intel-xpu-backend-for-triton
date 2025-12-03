
__kernel void slm_load_only_kernel(__global float* out, int iters)
{
    const int LOCAL_SIZE = 128;
    __local float smem[LOCAL_SIZE];

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    smem[lid] = (float)lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc = 0.0f;

    for (int i = 0; i < iters; ++i) {
        int idx = (lid + i) & (LOCAL_SIZE - 1);
        float v = smem[idx];
        acc += v;
    }

    out[gid] = acc;
}


__kernel void slm_store_only_kernel(__global float* out, int iters)
{
    const int LOCAL_SIZE = 128;
    __local float smem[LOCAL_SIZE];

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    smem[lid] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    float v = (float)lid;

    for (int i = 0; i < iters; ++i) {
        int idx = (lid + i) & (LOCAL_SIZE - 1);
        smem[idx] = v;
        v += 1.0f;
    }

    out[gid] = v;
}


__kernel void barrier_only_kernel(__global float* out, int iters)
{
    const int LOCAL_SIZE = 128;
    __local float smem[LOCAL_SIZE];

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    smem[lid] = lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    float tmp = 0.0f;

    for (int i = 0; i < iters; ++i) {
        barrier(CLK_LOCAL_MEM_FENCE);
        tmp += 0.0f;
    }

    out[gid] = tmp;
}
