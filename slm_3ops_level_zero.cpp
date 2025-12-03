#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>

#include <vector>
#include <algorithm>
#include <numeric>

struct Stats {
    double mean_ns;
    double median_ns;
};

Stats compute_stats_ns(std::vector<double> vals_ns) {
    std::sort(vals_ns.begin(), vals_ns.end());
    const int n = static_cast<int>(vals_ns.size());

    double sum = std::accumulate(vals_ns.begin(), vals_ns.end(), 0.0);
    double mean = sum / n;

    double median;
    if (n % 2 == 0) {
        median = 0.5 * (vals_ns[n/2 - 1] + vals_ns[n/2]);
    } else {
        median = vals_ns[n/2];
    }

    return {mean, median};
}


#define CHECK_ZE(call)                                                      \
    do {                                                                    \
        ze_result_t res = (call);                                           \
        if (res != ZE_RESULT_SUCCESS) {                                     \
            fprintf(stderr, "ZE ERROR %d at %s:%d\n",                       \
                    res, __FILE__, __LINE__);                               \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static std::vector<uint8_t> loadBinary(const char* file)
{
    std::ifstream f(file, std::ios::binary | std::ios::ate);
    if (!f) { perror("open fail"); exit(1); }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    if (!f.read((char*)buf.data(), size)) { perror("read fail"); exit(1); }
    return buf;
}

double run_kernel(ze_context_handle_t ctx,
                  ze_device_handle_t dev,
                  ze_command_queue_handle_t q,
                  ze_module_handle_t mod,
                  const char* name,
                  void* out_dev,
                  int iters,
                  uint32_t group_size,
                  uint32_t groups,
                  double timerResNs)
{
    // kernel create
    ze_kernel_desc_t kd = {};
    kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kd.pKernelName = name;

    ze_kernel_handle_t krnl;
    CHECK_ZE(zeKernelCreate(mod, &kd, &krnl));
    CHECK_ZE(zeKernelSetGroupSize(krnl, group_size, 1, 1));

    CHECK_ZE(zeKernelSetArgumentValue(krnl, 0, sizeof(out_dev), &out_dev));
    CHECK_ZE(zeKernelSetArgumentValue(krnl, 1, sizeof(iters), &iters));

    // cmdlist
    ze_command_list_desc_t ld = {};
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ld.commandQueueGroupOrdinal = 0;

    ze_command_list_handle_t cl;
    CHECK_ZE(zeCommandListCreate(ctx, dev, &ld, &cl));

    // event for kernel timestamps
    ze_event_pool_desc_t epd = {};
    epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    epd.count = 1;
    epd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

    ze_event_pool_handle_t pool;
    CHECK_ZE(zeEventPoolCreate(ctx, &epd, 1, &dev, &pool));

    ze_event_desc_t ed = {};
    ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    ed.index = 0;

    ze_event_handle_t evt;
    CHECK_ZE(zeEventCreate(pool, &ed, &evt));

    ze_group_count_t gc = { groups, 1, 1 };

    CHECK_ZE(zeCommandListAppendLaunchKernel(cl, krnl, &gc, evt, 0, nullptr));
    CHECK_ZE(zeCommandListClose(cl));
    CHECK_ZE(zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr));
    CHECK_ZE(zeCommandQueueSynchronize(q, UINT64_MAX));

    ze_kernel_timestamp_result_t ts = {};
    CHECK_ZE(zeEventQueryKernelTimestamp(evt, &ts));

    uint64_t ticks = ts.global.kernelEnd - ts.global.kernelStart;
    double ns = ticks * timerResNs;

    zeEventDestroy(evt);
    zeEventPoolDestroy(pool);
    zeCommandListDestroy(cl);
    zeKernelDestroy(krnl);

    return ns;
}
int main()
{
    constexpr int LOCAL_SIZE = 128;
    constexpr int GROUPS     = 1;
    constexpr int ITERS      = 100000;
    constexpr int REPEATS    = 1000;

    CHECK_ZE(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t dc = 0;
    CHECK_ZE(zeDriverGet(&dc, nullptr));
    std::vector<ze_driver_handle_t> drv(dc);
    CHECK_ZE(zeDriverGet(&dc, drv.data()));
    ze_driver_handle_t driver = drv[0];

    uint32_t devc = 0;
    CHECK_ZE(zeDeviceGet(driver, &devc, nullptr));
    std::vector<ze_device_handle_t> devs(devc);
    CHECK_ZE(zeDeviceGet(driver, &devc, devs.data()));
    ze_device_handle_t device = devs[0];

    ze_device_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    CHECK_ZE(zeDeviceGetProperties(device, &props));

    printf("Device: %s\n", props.name);
    double resNs = props.timerResolution;

    ze_context_handle_t ctx;
    ze_context_desc_t cd = { ZE_STRUCTURE_TYPE_CONTEXT_DESC };
    CHECK_ZE(zeContextCreate(driver, &cd, &ctx));

    ze_command_queue_desc_t qd = {};
    qd.stype  = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.ordinal = 0;
    qd.index   = 0;
    qd.mode    = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;

    ze_command_queue_handle_t q;
    CHECK_ZE(zeCommandQueueCreate(ctx, device, &qd, &q));

    // load module
    auto bin = loadBinary("slm_3ops_kernels_bmg.spv");

    ze_module_desc_t md = {};
    md.stype       = ZE_STRUCTURE_TYPE_MODULE_DESC;
    md.format      = ZE_MODULE_FORMAT_IL_SPIRV;
    md.inputSize   = bin.size();
    md.pInputModule = bin.data();

    ze_module_handle_t mod;
    ze_module_build_log_handle_t log;
    CHECK_ZE(zeModuleCreate(ctx, device, &md, &mod, &log));

    // allocate out buffer
    size_t nthreads = LOCAL_SIZE * GROUPS;
    size_t outBytes = nthreads * sizeof(float);

    ze_device_mem_alloc_desc_t da = {};
    da.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

    void* out_dev = nullptr;
    CHECK_ZE(zeMemAllocDevice(ctx, &da, outBytes, 64, device, &out_dev));

    // --- run 3 kernels multiple times and collect timings (ns) ---
    std::vector<double> times_load_ns;
    std::vector<double> times_store_ns;
    std::vector<double> times_barrier_ns;
    times_load_ns.reserve(REPEATS);
    times_store_ns.reserve(REPEATS);
    times_barrier_ns.reserve(REPEATS);

    for (int i = 0; i < REPEATS; ++i) {
        double ns_load =
            run_kernel(ctx, device, q, mod,
                       "slm_load_only_kernel",
                       out_dev, ITERS,
                       LOCAL_SIZE, GROUPS, resNs);

        double ns_store =
            run_kernel(ctx, device, q, mod,
                       "slm_store_only_kernel",
                       out_dev, ITERS,
                       LOCAL_SIZE, GROUPS, resNs);

        double ns_barrier =
            run_kernel(ctx, device, q, mod,
                       "barrier_only_kernel",
                       out_dev, ITERS,
                       LOCAL_SIZE, GROUPS, resNs);

        times_load_ns.push_back(ns_load);
        times_store_ns.push_back(ns_store);
        times_barrier_ns.push_back(ns_barrier);
    }

    // --- compute mean + median ---
    Stats load_stats    = compute_stats_ns(times_load_ns);
    Stats store_stats   = compute_stats_ns(times_store_ns);
    Stats barrier_stats = compute_stats_ns(times_barrier_ns);

    // per-op cost (no baseline!)
    double loads_per_kernel    = static_cast<double>(nthreads) * ITERS;
    double stores_per_kernel   = static_cast<double>(nthreads) * ITERS;
    double barriers_per_kernel = static_cast<double>(GROUPS)   * ITERS;

    double ns_per_load_mean      = load_stats.mean_ns    / loads_per_kernel;
    double ns_per_load_median    = load_stats.median_ns  / loads_per_kernel;

    double ns_per_store_mean     = store_stats.mean_ns   / stores_per_kernel;
    double ns_per_store_median   = store_stats.median_ns / stores_per_kernel;

    double ns_per_barrier_mean   = barrier_stats.mean_ns    / barriers_per_kernel;
    double ns_per_barrier_median = barrier_stats.median_ns  / barriers_per_kernel;

    // --- print results in the same style as CUDA version ---
    printf("=== Raw times over %d runs (ms) ===\n", REPEATS);
    printf("SLM load-only    : mean = %.3f ms, median = %.3f ms\n",
           load_stats.mean_ns   * 1e-6,
           load_stats.median_ns * 1e-6);
    printf("SLM store-only   : mean = %.3f ms, median = %.3f ms\n",
           store_stats.mean_ns   * 1e-6,
           store_stats.median_ns * 1e-6);
    printf("BARRIER-only     : mean = %.3f ms, median = %.3f ms\n",
           barrier_stats.mean_ns   * 1e-6,
           barrier_stats.median_ns * 1e-6);

    printf("\n=== Per-op times (ns) ===\n");
    printf("SLM LOAD per op   : mean = %.6f ns, median = %.6f ns\n",
           ns_per_load_mean, ns_per_load_median);
    printf("SLM STORE per op  : mean = %.6f ns, median = %.6f ns\n",
           ns_per_store_mean, ns_per_store_median);
    printf("BARRIER per op    : mean = %.6f ns, median = %.6f ns\n",
           ns_per_barrier_mean, ns_per_barrier_median);

    zeMemFree(ctx, out_dev);
    zeModuleDestroy(mod);
    zeCommandQueueDestroy(q);
    zeContextDestroy(ctx);

    return 0;
}
