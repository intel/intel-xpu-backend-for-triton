#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_epilogue.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_mma.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_softmax_epilogue.hpp"
#include "flash_attention_v2/kernel/tile_scheduler.hpp"
#include "flash_attention_v2/kernel/xe_flash_attn_prefill.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"

#include <exception>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// PRIVATE FUNCTION
////////////////////////////////////////////////////////////////////////////////

template <typename FMHA> static auto run(typename FMHA::Params params) -> void {
  cute::dim3 const block = FMHA::get_block_shape();
  cute::dim3 const grid = FMHA::get_grid_shape(params);

  int smem_size = FMHA::SharedStorageSize;

  const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
  using namespace syclcompat::experimental;
  auto event = launch<cutlass::device_kernel<FMHA>>(
      launch_policy{
          sycl_grid, sycl_block,
          local_mem_size{static_cast<std::size_t>(smem_size)},
          kernel_properties{
              sycl_exp::sub_group_size<FMHA::DispatchPolicy::SubgroupSize>}},
      params);
#else
  syclcompat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  syclcompat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<
          FMHA::DispatchPolicy::SubgroupSize>};
  syclcompat::experimental::launch_policy policy{sycl_grid, sycl_block,
                                                 launch_props, kernel_props};
  auto event = syclcompat::experimental::launch<cutlass::device_kernel<FMHA>>(
      policy, params);
#endif

  EventManager::getInstance().addEvent(event);
}

template <bool Causal, typename TileShapeQK, typename TileShapePV,
          typename TileShapeOutput, typename SubgroupLayout, int PipelineStages>
static auto attention_run(const at::Tensor &Q, const at::Tensor &K,
                          const at::Tensor &V, at::Tensor &O, int Batch,
                          int NumHeadsQ, int NumHeadsKV, int SeqLengthQO,
                          int SeqLengthKV, int HeadSizeQK, int HeadSizeVO,
                          float sm_scale) -> int {
  RECORD_FUNCTION("cutlass fa", {});

  using ElementAccumulator = float;
  using ElementInputQ = cutlass::half_t;
  using ElementInputKV = cutlass::half_t;
  using ElementOutput = float;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using MMAOperation = cute::XE_8x16x16_F32F16F16F32_TT;

  using GmemTiledCopyQ = cute::XE_2D_U16x8x32_LD_N;
  using GmemTiledCopyK = cute::XE_2D_U16x16x16_LD_T;
  using GmemTiledCopyV = cute::XE_2D_U16x16x32_LD_V;
  using GmemTiledCopyStore = cute::XE_2D_U32x8x16_ST_N;

  using ProblemShapeType = cute::tuple<int, int, int, int, int, int, int>;

  /// MAIN LOOP ///

  using CollectiveMainloop =
      cutlass::flash_attention::collective::FlashPrefillMma<
          GEMMDispatchPolicy, ProblemShapeType, ElementInputQ,
          cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
          cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV,
          cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation, TileShapeQK,
          TileShapePV, SubgroupLayout,
          GmemTiledCopyQ, // Q
          GmemTiledCopyK, // K
          GmemTiledCopyV, // V,
          Causal>;

  /// EPILOGUE LOOP ///

  using CollectiveSoftmaxEpilogue =
      cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<
          Causal, EpilogueDispatchPolicy, ElementAccumulator>;
  using CollectiveEpilogue =
      cutlass::flash_attention::collective::FlashPrefillEpilogue<
          EpilogueDispatchPolicy, MMAOperation, TileShapeOutput, SubgroupLayout,
          ElementAccumulator, cutlass::gemm::TagToStrideC_t<LayoutO>,
          ElementOutput, GmemTiledCopyStore>;

  /// FA ///

  using FMHAPrefillKernel = cutlass::flash_attention::kernel::FMHAPrefill<
      ProblemShapeType, CollectiveMainloop, CollectiveSoftmaxEpilogue,
      CollectiveEpilogue>;

  /// FA INVOCATION ///

  try {
    /// Buffer Initialization
    const cutlass::half_t *_Q =
        static_cast<const cutlass::half_t *>(Q.data_ptr());
    const cutlass::half_t *_K =
        static_cast<const cutlass::half_t *>(K.data_ptr());
    const cutlass::half_t *_V =
        static_cast<const cutlass::half_t *>(V.data_ptr());
    const float *_O = static_cast<const float *>(O.data_ptr());

    /// Problem size
    using ProblemShapeType = typename FMHAPrefillKernel::ProblemShape;
    ProblemShapeType problem_size =
        ProblemShapeType{Batch,       NumHeadsQ,  NumHeadsKV, SeqLengthQO,
                         SeqLengthKV, HeadSizeQK, HeadSizeVO};

    /// Stride
    using StrideQ = typename FMHAPrefillKernel::StrideQ;
    using StrideK = typename FMHAPrefillKernel::StrideK;
    using StrideV = typename FMHAPrefillKernel::StrideV;
    using StrideO = typename FMHAPrefillKernel::StrideO;
    StrideQ stride_Q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(SeqLengthQO, HeadSizeQK, Batch * NumHeadsQ));
    StrideK stride_K = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(SeqLengthKV, HeadSizeQK, Batch * NumHeadsKV));
    StrideV stride_V = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(HeadSizeVO, SeqLengthKV, Batch * NumHeadsKV));
    StrideO stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(SeqLengthQO, HeadSizeVO, Batch * NumHeadsQ));

    static cutlass::KernelHardwareInfo hw_info;
    if (hw_info.sm_count == 0) {
      hw_info.sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
      CUTLASS_TRACE_HOST(
          "Query result for SM count per device: " << hw_info.sm_count);
    }

    typename FMHAPrefillKernel::Arguments arguments = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {_Q, stride_Q, _K, stride_K, _V, stride_V},
        {sm_scale},
        {_O, stride_O},
        hw_info};

    size_t workspace_size = FMHAPrefillKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    auto workspace_ptr = workspace.get();

    if (!FMHAPrefillKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << Batch << 'x' << NumHeadsQ << 'x'
                << SeqLengthQO << 'x' << SeqLengthKV << 'x' << HeadSizeQK << 'x'
                << HeadSizeVO << (Causal ? "xCausal" : "xNonCausal")
                << std::endl;
      return -1;
    }

    CUTLASS_CHECK(
        FMHAPrefillKernel::initialize_workspace(arguments, workspace_ptr));
    auto params =
        FMHAPrefillKernel::to_underlying_arguments(arguments, workspace_ptr);
    run<FMHAPrefillKernel>(params);

    syclcompat::wait();

  } catch (std::exception &e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unexpected error" << std::endl;
    return -1;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// PUBLIC FUNCTION
////////////////////////////////////////////////////////////////////////////////

using FARunPtr = int (*)(const at::Tensor &Q, const at::Tensor &K,
                         const at::Tensor &V, at::Tensor &O, int Batch,
                         int NumHeadsQ, int NumHeadsKV, int SeqLengthQO,
                         int SeqLengthKV, int HeadSizeQK, int HeadSizeVO,
                         float sm_scale);

auto attention_kernel(const at::Tensor &Q, const at::Tensor &K,
                      const at::Tensor &V, at::Tensor &O, int Batch,
                      int NumHeadsQ, int NumHeadsKV, int SeqLengthQO,
                      int SeqLengthKV, int HeadSizeQK, int HeadSizeVO,
                      bool Causal, float sm_scale) -> int {
  constexpr int PipelineStages = 2;
  FARunPtr f = nullptr;

  if (HeadSizeVO == 64) {
    using ShapeQK = cute::Shape<cute::_128, cute::_64, cute::_64>;
    using ShapePV = cute::Shape<cute::_128, cute::_32, cute::_64>;
    using ShapeOutPut = cute::Shape<cute::_128, cute::_64, cute::_64>;
    using SubgroupLayout =
        cute::Layout<cute::Shape<cute::_8, cute::_1, cute::_1>,
                     cute::Stride<cute::_1, cute::_1, cute::_1>>;

    f = Causal ? attention_run<true, ShapeQK, ShapePV, ShapeOutPut,
                               SubgroupLayout, PipelineStages>
               : attention_run<false, ShapeQK, ShapePV, ShapeOutPut,
                               SubgroupLayout, PipelineStages>;

  } else if (HeadSizeVO == 128) {
    using ShapeQK = cute::Shape<cute::_128, cute::_64, cute::_64>;
    using ShapePV = cute::Shape<cute::_128, cute::_32, cute::_64>;
    using ShapeOutPut = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using SubgroupLayout =
        cute::Layout<cute::Shape<cute::_16, cute::_1, cute::_1>,
                     cute::Stride<cute::_1, cute::_1, cute::_1>>;

    f = Causal ? attention_run<true, ShapeQK, ShapePV, ShapeOutPut,
                               SubgroupLayout, PipelineStages>
               : attention_run<false, ShapeQK, ShapePV, ShapeOutPut,
                               SubgroupLayout, PipelineStages>;
  } else {
    std::cerr << "Unsupported HeadSizeVO: " << HeadSizeVO << std::endl;
    return -1;
  }

  return f(Q, K, V, O, Batch, NumHeadsQ, NumHeadsKV, SeqLengthQO, SeqLengthKV,
           HeadSizeQK, HeadSizeVO, sm_scale);
}
