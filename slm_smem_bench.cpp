#include <sycl/sycl.hpp>
#include <cstdio>

namespace sycl_ns = sycl;

constexpr int LOCAL_SIZE = 128;
constexpr int NUM_GROUPS = 1;
constexpr int N_ITERS    = 100000;

template <typename SubmitFn>
double run_and_time(sycl_ns::queue &q, SubmitFn submit_kernel) {
    sycl_ns::event e = submit_kernel(q);
    e.wait();
    uint64_t start =
        e.get_profiling_info<sycl_ns::info::event_profiling::command_start>();
    uint64_t end =
        e.get_profiling_info<sycl_ns::info::event_profiling::command_end>();
    return static_cast<double>(end - start);  // ns
}

int main() {
    try {
        sycl_ns::queue q{
            sycl_ns::gpu_selector_v,
            sycl_ns::property::queue::enable_profiling{}
        };

        std::printf("Running on device: %s\n",
                    q.get_device()
                        .get_info<sycl_ns::info::device::name>()
                        .c_str());

        const size_t NUM_THREADS = LOCAL_SIZE * NUM_GROUPS;
        sycl_ns::buffer<float, 1> out_buf{sycl_ns::range<1>(NUM_THREADS)};

        double t_load_ns = run_and_time(q, [&](sycl_ns::queue &qq) {
            return qq.submit([&](sycl_ns::handler &h) {
                auto out_acc =
                    out_buf.get_access<sycl_ns::access::mode::write>(h);

                sycl_ns::local_accessor<float, 1> smem{LOCAL_SIZE, h};

                h.parallel_for(
                    sycl_ns::nd_range<1>(
                        sycl_ns::range<1>(NUM_THREADS),
                        sycl_ns::range<1>(LOCAL_SIZE)
                    ),
                    [=](sycl_ns::nd_item<1> item) {
                        int lid = item.get_local_id(0);
                        int gid = item.get_global_id(0);

                        // init SLM + sync once
                        smem[lid] = (float)lid;
                        item.barrier(sycl_ns::access::fence_space::local_space);

                        float acc = 0.0f;

                        for (int i = 0; i < N_ITERS; ++i) {
                            int idx = (lid + i) & (LOCAL_SIZE - 1);
                            float v = smem[idx];
                            acc += v;
                        }

                        out_acc[gid] = acc;
                    }
                );
            });
        });

        double t_store_ns = run_and_time(q, [&](sycl_ns::queue &qq) {
            return qq.submit([&](sycl_ns::handler &h) {
                auto out_acc =
                    out_buf.get_access<sycl_ns::access::mode::write>(h);

                sycl_ns::local_accessor<float, 1> smem{LOCAL_SIZE, h};

                h.parallel_for(
                    sycl_ns::nd_range<1>(
                        sycl_ns::range<1>(NUM_THREADS),
                        sycl_ns::range<1>(LOCAL_SIZE)
                    ),
                    [=](sycl_ns::nd_item<1> item) {
                        int lid = item.get_local_id(0);
                        int gid = item.get_global_id(0);

                        smem[lid] = 0.0f;
                        item.barrier(sycl_ns::access::fence_space::local_space);

                        float v = (float)lid;

                        for (int i = 0; i < N_ITERS; ++i) {
                            int idx = (lid + i) & (LOCAL_SIZE - 1);
                            smem[idx] = v; 
                            v += 1.0f;
                        }

                        out_acc[gid] = v;
                    }
                );
            });
        });

        double t_barrier_ns = run_and_time(q, [&](sycl_ns::queue &qq) {
            return qq.submit([&](sycl_ns::handler &h) {
                auto out_acc =
                    out_buf.get_access<sycl_ns::access::mode::write>(h);

                sycl_ns::local_accessor<float, 1> smem{LOCAL_SIZE, h};

                h.parallel_for(
                    sycl_ns::nd_range<1>(
                        sycl_ns::range<1>(NUM_THREADS),
                        sycl_ns::range<1>(LOCAL_SIZE)
                    ),
                    [=](sycl_ns::nd_item<1> item) {
                        int lid = item.get_local_id(0);
                        int gid = item.get_global_id(0);

                        smem[lid] = (float)lid;
                        item.barrier(sycl_ns::access::fence_space::local_space);

                        float acc = 0.0f;

                        for (int i = 0; i < N_ITERS; ++i) {
                            item.barrier(
                                sycl_ns::access::fence_space::local_space);
                           acc += 0.0f;
                        }

                        out_acc[gid] = acc;
                    }
                );
            });
        });
--

        double num_threads      = static_cast<double>(NUM_THREADS);
        double num_iters        = static_cast<double>(N_ITERS);
        double loads_per_kernel = num_threads * num_iters;
        double stores_per_kernel = num_threads * num_iters;
        double barriers_per_kernel = NUM_GROUPS * num_iters;

        double ns_per_load    = t_load_ns    / loads_per_kernel;
        double ns_per_store   = t_store_ns   / stores_per_kernel;
        double ns_per_barrier = t_barrier_ns / barriers_per_kernel;

        std::printf("=== Raw times (ms) ===\n");
        std::printf("load-only      : %.3f ms\n", t_load_ns    * 1e-6);
        std::printf("store-only     : %.3f ms\n", t_store_ns   * 1e-6);
        std::printf("barrier-only   : %.3f ms\n", t_barrier_ns * 1e-6);

        std::printf("\n=== Per-op times (ns), no baseline ===\n");
        std::printf("SLM load       : %.6f ns\n", ns_per_load);
        std::printf("SLM store      : %.6f ns\n", ns_per_store);
        std::printf("barrier(sync)  : %.6f ns\n", ns_per_barrier);

        {
            sycl_ns::host_accessor host_acc(out_buf, sycl_ns::read_only);
            float checksum = 0.0f;
            for (size_t i = 0; i < NUM_THREADS; ++i)
                checksum += host_acc[i];
            std::printf("\nChecksum (ignore): %.3f\n", checksum);
        }

    } catch (const sycl_ns::exception &e) {
        std::fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }

    return 0;
}
