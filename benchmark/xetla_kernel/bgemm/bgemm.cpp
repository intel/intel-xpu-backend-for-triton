//Enable the FLAG when testing ESIMD
#define XETLA_CODE_BASE __ESIMD__
#define ESIMD_XE_HPC
#include "bgemm.h"
#include "../tests/utils/buff_compare.hpp"
#include "../tests/utils/gemm_gen.hpp"
#include "../tests/utils/profiling.hpp"
#include "kernel_func.hpp"
#include "test.hpp"
#include <gtest/gtest.h>

#ifdef _WIN32
#include "tests/utils/windows_functions.hpp"
#endif

using namespace gpu::xetla;

using namespace cl::sycl;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        size_t m, size_t k, size_t n,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);
    bool result = buff_cmp::xetla_buff_cmp(data, other, "bgemm_perf");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}
//Test: accept different test data
//mode: distinguish CM and ESIMD paths
//iter: indicate the iterations of the kernel
template <class Test>
void bgemm_run(ExecutionMode mode, int iter) {
    //Accept incoming parameters
    size_t matrix_m = Test::mat_m;
    size_t matrix_n = Test::mat_n;
    size_t matrix_k = Test::mat_k;
    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;
    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_acc = float;
    using bgemm_functor = bgemm_test_func<data_type_a, data_type_b, data_type_c,
            data_type_acc, wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n,
            sg_tile_k, Test::layout_a, Test::layout_b, Test::global_kslicing,
            Test::local_kslicing>;
    using gemm_op_t = typename bgemm_functor::gemm_op_t;

    size_t lda = Test::layout_a == mem_layout::col_major ? matrix_m : matrix_k;
    size_t ldb = Test::layout_b == mem_layout::col_major ? matrix_k : matrix_n;
    size_t ldc = matrix_n;

    std::string mem_layout_a_str = Test::layout_a == mem_layout::col_major
            ? "gpu::xetla::mem_layout::col_major"
            : "gpu::xetla::mem_layout::row_major";
    std::string mem_layout_b_str = Test::layout_b == mem_layout::col_major
            ? "gpu::xetla::mem_layout::col_major"
            : "gpu::xetla::mem_layout::row_major";

    constexpr bool is_col_major_a = Test::layout_a == mem_layout::col_major;
    constexpr bool is_col_major_b = Test::layout_b == mem_layout::col_major;

    size_t size_a = matrix_m * matrix_k;
    size_t size_b = matrix_k * matrix_n;
    size_t size_c = matrix_m * matrix_n;
    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    //Define and initialize the data required for the calculation
    data_type_a *A = static_cast<data_type_a *>(
            malloc_shared(size_a * sizeof(data_type_a), device, context));
    data_type_b *B = static_cast<data_type_b *>(
            malloc_shared(size_b * sizeof(data_type_b), device, context));
    data_type_c *C = static_cast<data_type_c *>(
            malloc_shared(size_c * sizeof(data_type_c), device, context));
    data_type_acc *Acc = static_cast<data_type_acc *>(
            malloc_shared(size_acc * sizeof(data_type_acc), device, context));
    uint32_t *Cnt = static_cast<uint32_t *>(
            malloc_shared(size_cnt * sizeof(uint32_t), device, context));

    for (size_t i = 0; i < size_a; ++i) {
        A[i] = (random_float() - 0.5f);
    }
    for (size_t i = 0; i < size_b; ++i) {
        B[i] = (random_float() - 0.5f);
    }
    for (size_t i = 0; i < size_c; ++i) {
        C[i] = 0;
    }
    for (size_t i = 0; i < size_acc; ++i) {
        Acc[i] = 0;
    }
    for (size_t i = 0; i < size_cnt; ++i) {
        Cnt[i] = 0;
    }
    // here keep the same dim in CM and esimd, diff the index in kernel code
    size_t group_range_m = (matrix_m % wg_tile_m == 0)
            ? matrix_m / wg_tile_m
            : (matrix_m / wg_tile_m) + 1;
    size_t group_range_n = (matrix_n % wg_tile_n == 0)
            ? matrix_n / wg_tile_n
            : (matrix_n / wg_tile_n) + 1;
    size_t subgroup_range_m = (wg_tile_m % sg_tile_m == 0)
            ? wg_tile_m / sg_tile_m
            : (wg_tile_m / sg_tile_m) + 1;
    size_t subgroup_range_n = (wg_tile_n % sg_tile_n == 0)
            ? wg_tile_n / sg_tile_n
            : (wg_tile_n / sg_tile_n) + 1;
    std::cout << "group_num_x: " << group_range_n
              << ", group_num_y: " << group_range_m
              << ", group_num_z: " << Test::global_kslicing << "\n";
    std::cout << "group_size_x: " << subgroup_range_n
              << ", group_size_y: " << subgroup_range_m
              << ", group_size_z: " << Test::local_kslicing << std::endl;
    cl::sycl::range<3> group_range {
            Test::global_kslicing, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {
            Test::local_kslicing, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    long ops = 2 * matrix_m * matrix_n * matrix_k;
    profiling_helper prof("bgemm", ops, "gflops");

    // esimd kernel prepratation and execution
    if (mode == ExecutionMode::ESIMD) {
        std::vector<kernel_id> kernelId = {get_kernel_id<Test>()};
        auto inputBundle
                = get_kernel_bundle<bundle_state::input>(context, kernelId);
        setenv("SYCL_PROGRAM_COMPILE_OPTIONS",
                " -vc-codegen -doubleGRF -vc-disable-indvars-opt "
                " -Xfinalizer '-printregusage -enableBCR -DPASTokenReduction '",
                1);
        kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
        unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");

        try {
            for (int i = 0; i < iter; i++) {
                prof.cpu_start();
                auto e_esimd = queue.submit([&](handler &cgh) {
                    cgh.use_kernel_bundle(exeBundle);
                    cgh.parallel_for<Test>(
                            nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                                constexpr uint32_t barrier_count
                                        = bgemm_functor::barrier_count;
                                constexpr uint32_t slm_size
                                        = bgemm_functor::slm_size;
                                if constexpr (barrier_count != 0) {
                                    xetla_nbarrier_init<barrier_count>();
                                }
                                if constexpr (slm_size != 0) {
                                    xetla_local_init<slm_size>();
                                }
                                bgemm_functor::run(item, A, B, C, matrix_m,
                                        matrix_n, matrix_k, lda, ldb, ldc, Acc,
                                        Cnt);
                            });
                });
                e_esimd.wait();
                prof.cpu_end();
                prof.add_gpu_event(e_esimd);
            }
        } catch (cl::sycl::exception const &e) {
            std::cout << "SYCL exception caught: " << e.what() << '\n';
            FAIL();
        }
    }

    prof.print_profiling_result(profiling_selector::GPU);

    // validation
    int err_cnt;
    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, matrix_m, matrix_k, matrix_n,
                    Test::layout_a, Test::layout_b));

    free(A, context);
    free(B, context);
    free(C, context);
    free(Acc, context);
    free(Cnt, context);
}

// template <typename T>
// class bgemm_performance_test : public ::testing::Test {};
// TYPED_TEST_SUITE_P(bgemm_performance_test);
// TYPED_TEST_P(bgemm_performance_test, esimd) {
//     bgemm_run<TypeParam>(ExecutionMode::ESIMD, ITER);
// }
// REGISTER_TYPED_TEST_SUITE_P(bgemm_performance_test, esimd);
// using tests =
//     ::testing::Types<Test_4096x4096x4096_row_row, Test_3968x3968x3968_row_row,
//                      Test_3840x3840x3840_row_row, Test_3712x3712x3712_row_row,
//                      Test_3584x3584x3584_row_row, Test_3456x3456x3456_row_row,
//                      Test_3328x3328x3328_row_row, Test_3200x3200x3200_row_row,
//                      Test_3072x3072x3072_row_row, Test_2944x2944x2944_row_row,
//                      Test_2816x2816x2816_row_row, Test_2688x2688x2688_row_row,
//                      Test_2560x2560x2560_row_row, Test_2432x2432x2432_row_row,
//                      Test_2304x2304x2304_row_row, Test_2176x2176x2176_row_row,
//                      Test_2048x2048x2048_row_row, Test_1920x1920x1920_row_row,
//                      Test_1792x1792x1792_row_row, Test_1664x1664x1664_row_row,
//                      Test_1536x1536x1536_row_row, Test_1408x1408x1408_row_row,
//                      Test_1280x1280x1280_row_row, Test_1152x1152x1152_row_row,
//                      Test_1024x1024x1024_row_row, Test_896x896x896_row_row,
//                      Test_768x768x768_row_row, Test_512x512x512_row_row,
//                      Test_640x640x640_row_row, Test_384x384x384_row_row,
//                      Test_256x256x256_row_row>;
// INSTANTIATE_TYPED_TEST_SUITE_P(
//         bgemm_performance_test_suite, bgemm_performance_test, tests);

template void bgemm_run<Test_4096x4096x4096_row_row>(int mode, const int64_t iter);