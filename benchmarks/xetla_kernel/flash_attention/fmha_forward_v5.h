/*******************************************************************************
 * Copyright (c) 2023-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef TRITONBENCHMARK_FMHA_FWD_V5_H
#define TRITONBENCHMARK_FMHA_FWD_V5_H

#include "fmha_policy_v2.h"
#include "fmha_utils.h"
#include "xetla.hpp"

// Set to 1 to get raw output, not permuted
#define _RAW_OUTPUT 0
#define _USE_BFNH 1
using T = sycl::ext::oneapi::bfloat16;

namespace gpu::xetla {

namespace fmha {

struct Shape {
  Shape(int B, int N, int F, int T, int H)
      : num_batches(B), num_heads(N), num_queries(F), num_keys(T),
        head_size(H) {}
  const int num_batches;
  const int num_heads;
  const int num_queries;
  const int num_keys;
  const int head_size;

  inline uint32_t get_query_size() const {
    return num_batches * num_heads * num_queries * head_size;
  }
  inline uint32_t get_key_size() const {
    return num_batches * num_heads * num_keys * head_size;
  }
  inline uint32_t get_score_size() const {
    return num_batches * num_heads * num_queries * num_keys;
  }
  inline uint32_t get_ml_size() const {
    return num_batches * num_heads * num_queries;
  }
  inline uint32_t get_attn_mask_size() const {
#if _BIAS_AS_INPUT
    return num_batches * num_heads * num_queries * num_keys;
#else
    return num_batches * num_queries * num_keys;
#endif
  }
};

template <typename fmha_policy, typename scalar_t, bool kUseBias,
          bool kIsCausal, bool kIsTraining>
class fmha_forward_t {
public:
  using accum_t = float;
  static constexpr accum_t kNegInfinity = INFINITY * -1;

  struct arguments_t {
    // Input tensors
    scalar_t *Q_ptr;           // [B, F, N, H] -> query
    scalar_t *K_ptr;           // [B, T, N, H] -> key
    scalar_t *V_ptr;           // [B, T, N, H] -> value
    scalar_t *B_ptr = nullptr; // [B, 1/N, F, T] - bias
    // Dropout scale is computed from dropout prob
    accum_t dp_prob;
    accum_t dp_scale;
    // Output tensor
    scalar_t *O_ptr; // permute: [B, F, N, H] - output
    accum_t *L_ptr;  // [B, N, F]
    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uH;
    uint32_t uF;
    uint32_t uT;
    // Softmax scale is the reciprocal square root of head size by default
    accum_t sm_scale;
    // seed/offset used to generate dropout mask
    uint64_t seed;
    uint64_t offset;

    inline arguments_t() = default;
    inline arguments_t(scalar_t *query, scalar_t *key, scalar_t *value,
                       scalar_t *bias, accum_t dropout_prob, scalar_t *out,
                       accum_t *l, uint32_t num_batches, uint32_t num_heads,
                       uint32_t head_size, uint32_t num_queries,
                       uint32_t num_keys, accum_t sm_scale, uint64_t seed,
                       uint64_t offset)
        : Q_ptr(query), K_ptr(key), V_ptr(value), B_ptr(bias),
          dp_prob(dropout_prob), dp_scale(1.f / (1.f - dropout_prob)),
          O_ptr(out), L_ptr(l), uB(num_batches), uN(num_heads), uH(head_size),
          uF(num_queries), uT(num_keys), sm_scale(sm_scale), seed(seed),
          offset(offset) {}
  };

private:
  // -------------------- // Compute policy // -------------------- //
  static constexpr uint32_t accum_step = fmha_policy::accum_step;
  static constexpr uint32_t stages = fmha_policy::stages;
  static constexpr uint32_t sync_freq = fmha_policy::sync_freq;

  using compute_attr = group::compute_attr_t<scalar_t, scalar_t, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<accum_step, stages, sync_freq>;
  using compute_policy =
      group::compute_policy_default_xmx<compute_attr, perf_tuning_knob,
                                        gpu_arch::Xe>;

  // ---------------- // Tile shape and Threads // ---------------- //
  static constexpr uint32_t kBr = fmha_policy::kBr;
  static constexpr uint32_t kBc = fmha_policy::kBc;
  static constexpr uint32_t kHm = fmha_policy::kHm;
  static constexpr uint32_t kSgBr = fmha_policy::kSgBr;
  static constexpr uint32_t kSgBc = fmha_policy::kSgBc;
  static constexpr uint32_t kSgHm = fmha_policy::kSgHm;

  using tile_shape_BrBc = group::tile_shape_t<kBc, kBr, kSgBc, kSgBr>;
  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;

  static constexpr uint32_t wg_size_x = tile_shape_BrBc::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrBc::wg_size_y;
  using work_group_t = typename tile_shape_BrBc::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static_assert(kHm / kSgHm == kBc / kSgBc,
                "wg_size_x must be the same between Hm and Bc");
  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  // --------------------- // Memory desc // ---------------------- //
  // suffix: L -> local; T -> transpose
  using mem_desc_Qi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Qi_L_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_Kj_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_Pij_L_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_Vj_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Bij_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Oi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Mi_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Li_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Dp_Mask_t =
      mem_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;

  // ------------------- // Slm and nbarrier // ------------------- //
  static constexpr uint32_t slm_size_Qi = kBr * kHm * sizeof(scalar_t);
  static constexpr uint32_t slm_size_Pij = kBr * kBc * sizeof(scalar_t);
  static constexpr uint32_t slm_size_softmax =
      (wg_size_x > 1) ? wg_size * kSgBr * sizeof(accum_t) : 0;
  // Slm addr to store inermediate results
  static constexpr uint32_t Qi_slm = 0;
  static constexpr uint32_t Pij_slm = Qi_slm + slm_size_Qi;
  static constexpr uint32_t softmax_slm = Pij_slm + slm_size_Pij;

  static constexpr uint32_t nbarrier_cnt = (wg_size_x > 1) ? wg_size_y : 0;
  // Define kernel to compute Sij = Qi x Kj.T
  using brgemm_Sij_t = group::gemm_t<compute_policy, tile_shape_BrBc,
                                     mem_desc_Qi_L_t, mem_desc_Kj_T_t>;
  using matAccSij_t = typename brgemm_Sij_t::matAcc_t;
  using dropout_t = dropout_fwd_t<matAccSij_t::tile_elems>;
  // ======================== // Context // ======================= //

  /// @brief Used to store variables in the flash mha loops
  struct context_t {
    // thread id
    work_group_t g;
    uint32_t sg_idx;
    uint32_t sg_idy;
    // nbarrier
    xetla_nbarrier_t<wg_size_x, wg_size_x> nbarrier;
    // softmax statistics
    xetla_vector<accum_t, kSgBr> softmax_m;
    xetla_vector<accum_t, kSgBr> softmax_l;
    // mem desc variables
    mem_desc_Qi_t mem_desc_Qi;
    mem_desc_Qi_L_t mem_desc_Qi_L;
    mem_desc_Kj_T_t mem_desc_Kj_T;
    mem_desc_Pij_L_t mem_desc_Pij_L;
    mem_desc_Vj_t mem_desc_Vj;
    mem_desc_Bij_t mem_desc_Bij;
    mem_desc_Oi_t mem_desc_Oi;
    mem_desc_Mi_t mem_desc_Mi;
    mem_desc_Li_t mem_desc_Li;
    dropout_t dropout_op;

    inline context_t() = default;

    /// @brief Initialize invariant variables in the flash mha loop
    inline void init_context(const sycl::nd_item<3> &ei,
                             const arguments_t &args) {
      // thread id
      uint32_t sg_id = ei.get_local_linear_id();
      g.init(sg_id);
      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      // nbarrier
      nbarrier.init_nbarrier(sg_idy, nbarrier_role::producer_consumer);
      // softmax statistics
      softmax_m = kNegInfinity;
      softmax_l = 0.f;

      // mem desc variables
      uint32_t gid = ei.get_group(0);
      uint32_t bid = ei.get_group(0) / args.uN;
      uint32_t nid = ei.get_group(0) % args.uN;
      int32_t start_x = nid * args.uH;
      uint32_t end_x = start_x + args.uH;
      int32_t start_y = bid * args.uF + ei.get_group(1) * kBr;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (bid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;
      uint32_t pitch = args.uH * args.uN;

      mem_desc_Qi.init(args.Q_ptr, {end_x, end_y, pitch}, {start_x, start_y});
      mem_desc_Oi.init(args.O_ptr, {end_x, end_y, pitch}, {start_x, start_y});

      int32_t start_x_ml = ei.get_group(1) * kBr + sg_idy * kSgBr;
      int32_t start_y_ml = gid;
      mem_desc_Li.init(args.L_ptr, {args.uF, args.uB * args.uN, args.uF},
                       {start_x_ml, start_y_ml});

      mem_desc_Qi_L.init(Qi_slm, {kHm, kBr, kHm}, {0, 0});
      mem_desc_Pij_L.init(Pij_slm, {kBc, kBr, kBc}, {0, 0});

      if constexpr (kIsTraining) {
        uint64_t sg_subseq = ei.get_global_linear_id();
        dropout_op.init(args.seed, sg_subseq, args.offset, args.dp_prob,
                        args.dp_scale);
      }
    }

    /// @brief Update variables for each flash mha loop
    inline void update_context(const sycl::nd_item<3> &ei,
                               const arguments_t &args, uint32_t startT) {
      uint32_t gid = ei.get_group(0);
      uint32_t bid = gid / args.uN;
      uint32_t nid = gid % args.uN;

      int32_t start_x = bid * args.uT + startT;
      uint32_t end_x = start_x + kBc;
      uint32_t boundary_x = (bid + 1) * args.uT;
      end_x = end_x > boundary_x ? boundary_x : end_x;
      int32_t start_y = nid * args.uH;
      uint32_t end_y = start_y + args.uH;

      uint32_t pitch = args.uN * args.uH;

      mem_desc_Kj_T.init(args.K_ptr, {end_x, end_y, pitch}, {start_x, start_y});
      mem_desc_Vj.init(args.V_ptr, {end_y, end_x, pitch}, {start_y, start_x});

      if constexpr (kUseBias) {
        start_x = startT;
        end_x = start_x + kBc;
        boundary_x = args.uT;
        end_x = end_x > boundary_x ? boundary_x : end_x;

        int32_t start_y = bid * args.uF + ei.get_group(1) * kBr;
        uint32_t end_y = start_y + kBr;
        uint32_t boundary_y = (bid + 1) * args.uF;
        end_y = end_y > boundary_y ? boundary_y : end_y;

        mem_desc_Bij.init(args.B_ptr, {end_x, end_y, args.uT},
                          {start_x, start_y});
      }
    }
  };

  context_t ctx;

  // ======================= // gemm_Sij // ======================= //

  /// @brief gemm_Sij is used to compute Sij = Qi x Kj.T
  /// # [Br,H] x [H,Bc] = [Br,Bc]
  inline void gemm_Sij(matAccSij_t *matAccSij, const arguments_t &args) {
    using brgemm_args_t = typename brgemm_Sij_t::arguments_t;

    // Gemm to comput Sij
    brgemm_Sij_t brgemm;
    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;
    brgemm_args_t brgemm_args(ctx.mem_desc_Qi_L, ctx.mem_desc_Kj_T, loop_count);
    brgemm(ctx.g, *matAccSij, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);

    // Multiply by softmax scaling factor
    matAccSij->reg *= args.sm_scale;

    // Add bias if needed
    if constexpr (kUseBias) {
      using bias_op_t = subgroup::elemwise_reduce_op_t<reduce_op::sum, scalar_t,
                                                       gpu_arch::Xe>;
      using bias_args_t = typename bias_op_t::arguments_t;

      int32_t tile_offset_x = ctx.sg_idx * kSgBc;
      int32_t tile_offset_y = ctx.sg_idy * kSgBr;
      ctx.mem_desc_Bij.update_coord(tile_offset_x, tile_offset_y);

      bias_op_t bias_op;
      bias_args_t bias_args(ctx.mem_desc_Bij.base, ctx.mem_desc_Bij.shape);
      bias_op(*matAccSij, ctx.mem_desc_Bij.coord, bias_args);
    }
  }

  // ======================= // gemm_Oi // ======================= //
  // Define kernel to compute Oi += Pij x Vj
  using brgemm_Oi_t = group::gemm_t<compute_policy, tile_shape_BrHm,
                                    mem_desc_Pij_L_t, mem_desc_Vj_t>;
  using matAccOi_t = typename brgemm_Oi_t::matAcc_t;

  /// @brief gemm_Oi is used to compute Oi += Pij x Vj
  /// # [Br,Bc] x [Bc,H] = [Br,Hm]
  inline void gemm_Oi(matAccOi_t *matAccOi, const arguments_t &args,
                      uint32_t startT) {
    using brgemm_args_t = typename brgemm_Oi_t::arguments_t;

    uint32_t remainT = args.uT - startT;
    uint32_t boundary_k = remainT > kBc ? kBc : remainT;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to comput Oi
    brgemm_Oi_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_Pij_L, ctx.mem_desc_Vj, loop_count);
    brgemm(ctx.g, *matAccOi, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);
  }

  // ====================== // apply_mask // ====================== //

  /// @brief apply mask to matAccSij.
  inline void apply_mask(matAccSij_t *matAccSij, const arguments_t &args,
                         uint32_t startF, uint32_t startT) {
    using tile_mask = tile_mask_t<matAccSij_t>;

    uint32_t sg_startT = startT + ctx.sg_idx * kSgBc;
    uint32_t remainT = std::max(int(args.uT) - int(sg_startT), 0);
    if (remainT < kSgBc) {
      tile_mask::padding_mask(matAccSij, remainT);
    }

    if constexpr (kIsCausal) {
      uint32_t sg_startF = startF + ctx.sg_idy * kSgBr;
      if (sg_startT + kSgBc > sg_startF) {
        tile_mask::causal_mask(matAccSij, sg_startT, sg_startF);
      }
    }
  }

  // ====================== // softmax_fwd // ===================== //

  /// @brief softmax_fwd is used to do softmax.
  inline void softmax_fwd(matAccSij_t *matAccSij, matAccOi_t *matAccOi,
                          const arguments_t &args) {
    using wg_row_max_t =
        group_row_reduce_t<matAccSij_t, wg_size_x, reduce_op::max>;
    using wg_row_sum_t =
        group_row_reduce_t<matAccSij_t, wg_size_x, reduce_op::sum>;

    // init slm address for group reducer
    uint32_t reducer_slm =
        softmax_slm + ctx.sg_idy * wg_size_x * kSgBr * sizeof(accum_t);

    // compute new m
    wg_row_max_t wg_row_max(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> m_new = wg_row_max(matAccSij);
    m_new = xetla_max<accum_t, kSgBr>(m_new, ctx.softmax_m);

    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive();

    // correct old l
    ctx.softmax_l *= xetla_exp<accum_t, kSgBr>(ctx.softmax_m - m_new);
    // compute Pij
    subgroup::tile_broadcast_op<subgroup::tile_minus, matAccSij_t>(*matAccSij,
                                                                   m_new);
    matAccSij->reg = xetla_exp<accum_t>(matAccSij->reg);

    if constexpr (wg_size_x > 1)
      ctx.nbarrier.wait();

    // compute new l
    wg_row_sum_t wg_row_sum(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> l_new = wg_row_sum(matAccSij);
    l_new += ctx.softmax_l;

    // rescale operands of matmuls
    subgroup::tile_broadcast_op<subgroup::tile_div, matAccSij_t>(*matAccSij,
                                                                 l_new);
    xetla_vector<accum_t, kSgBr> o_scale = l_new / ctx.softmax_l;
    subgroup::tile_broadcast_op<subgroup::tile_div, matAccOi_t>(*matAccOi,
                                                                o_scale);
    // update m and l for the next step
    ctx.softmax_m = m_new;
    ctx.softmax_l = l_new;

    if constexpr (kIsTraining) {
      matAccSij->reg = ctx.dropout_op.template process<float>(matAccSij->reg);
    }

    // save Pij to local memory
    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrBc, mem_desc_Pij_L_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, *matAccSij, ctx.mem_desc_Pij_L);
    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
  }

  // ==================== // store_Oi // ====================== //

  /// @brief store raw Oi to global memory. [B,N,F,H]
  inline void store_Oi(matAccOi_t &matAccOi, const arguments_t &args) {
    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrHm, mem_desc_Oi_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matAccOi, ctx.mem_desc_Oi);
  }

  inline void store_for_backward(const arguments_t &args) {
    // save m and l to global
    if constexpr (!kIsTraining) {
      return;
    }
    using store_desc =
        subgroup::tile_desc_t<kSgBr, 1, kSgBr, 1, reg_layout::tiled>;
    using store_tile_t = subgroup::tile_t<accum_t, store_desc>;
    // Note: use block_2d store as only block_2d supports boundary check
    using store_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        store_desc, msg_type::block_2d, gpu_arch::Xe>;
    store_tile_t mat_store;
    store_payload_t store_payload(ctx.mem_desc_Li);
    mat_store.reg =
        ctx.softmax_m + sycl::ext::intel::esimd::log2(ctx.softmax_l);
    if (ctx.sg_idx == 0) {
      subgroup::tile_store(mat_store, store_payload);
    }
  }

  // ================== // permute_store_Oi // ==================== //

  /// @brief permuted store Oi to global memory. [B,F,N,H]
  inline void permute_store_Oi(const sycl::nd_item<3> &ei, matAccOi_t *matAccOi,
                               const arguments_t &args) {
    int b = ei.get_group(0) / args.uN;
    int n = ei.get_group(0) % args.uN;
    int f = ctx.sg_idy * kSgBr + ei.get_group(1) * kBr;
    int h = ctx.sg_idx * kSgHm;

    // Because Hm is greater than uH
    if (h >= args.uH)
      return;

    xetla_tdescriptor transpose_tdecs;
    xetla_vector<scalar_t, kSgHm> v_out;

    int height = args.uB * args.uN * args.uF;
    // TODO(zw): change to b if compiler issue is fixed
    int offset_height =
        (ei.get_group(0) / args.uN) * args.uN * args.uF + f * args.uN + n;

    xetla_fill_tdesc<scalar_t, kSgHm, 1, 1>(
        transpose_tdecs.xetla_format<uint32_t>(), args.O_ptr, args.uH, height,
        args.uH, h, offset_height);

    using load_t = load_tile_t<matAccOi_t>;
    for (uint32_t i = 0; i < kSgBr && (f + i < args.uF); ++i) {
      // load data from matAccOi
      xetla_vector<accum_t, kSgHm> v_acc;
      load_t::load_tile_x(matAccOi, i, v_acc);
      v_out = xetla_cvt<scalar_t, accum_t, kSgHm>(v_acc);

      xetla_tstore_global<scalar_t, kSgHm, cache_hint::write_back,
                          cache_hint::write_back>(transpose_tdecs, v_out);
      xetla_update_tdesc_offsety(transpose_tdecs.xetla_format<uint32_t>(),
                                 args.uN);
    }
  }

  // ====================== // preload_Qi // ====================== //

  /// @brief preload_Qi is used to load Qi from global to local memory.
  inline void preload_Qi(const arguments_t &args) {
    using matQi_tile_desc_t = typename brgemm_Oi_t::matAcc_tile_desc_t;
    using matQi_t = subgroup::tile_t<scalar_t, matQi_tile_desc_t>;
    using matQi_load_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_desc_Qi_t::layout, mem_desc_Qi_t::space>,
        matQi_tile_desc_t,
        subgroup::msg_type_v<matQi_tile_desc_t, mem_desc_Qi_t::space>,
        gpu_arch::Xe>;
    using matQi_store_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_desc_Qi_L_t::layout, mem_desc_Qi_L_t::space>,
        matQi_tile_desc_t,
        subgroup::msg_type_v<matQi_tile_desc_t, mem_desc_Qi_L_t::space>,
        gpu_arch::Xe>;

    int32_t tile_offset_x = ctx.sg_idx * kSgHm;
    int32_t tile_offset_y = ctx.sg_idy * kSgBr;

    mem_desc_Qi_t mem_desc_Qi_load(ctx.mem_desc_Qi);
    mem_desc_Qi_L_t mem_desc_Qi_store(ctx.mem_desc_Qi_L);

    mem_desc_Qi_load.update_coord(tile_offset_x, tile_offset_y);
    mem_desc_Qi_store.update_coord(tile_offset_x, tile_offset_y);

    matQi_t matQi;
    matQi_load_t matQi_load(mem_desc_Qi_load);
    subgroup::tile_load(matQi, matQi_load);

    matQi_store_t matQi_store(mem_desc_Qi_store);
    subgroup::tile_store(matQi, matQi_store);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
  }

public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_Sij = brgemm_Sij_t::barrier_count;
    constexpr uint32_t barrier_count_Oi = brgemm_Oi_t::barrier_count;
    constexpr uint32_t count =
        std::max(barrier_count_Sij, barrier_count_Oi) + nbarrier_cnt;
    static_assert(count <= 32,
                  "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_Qi + slm_size_Pij + slm_size_softmax;
    static_assert(size <= (128 * 1024),
                  "The local memory size should be less than 128KB!");
    return size;
  };

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(uint32_t total_batches,
                                        uint32_t num_queries) {
    // local range
    sycl::range<3> local_range = sycl::range<3>{1, wg_size_y, wg_size_x};
    // group range
    uint32_t group_range_m = (num_queries + kBr - 1) / kBr;
    sycl::range<3> group_range =
        sycl::range<3>{total_batches, group_range_m, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  // ================= // Entry of the functor // ================= //

  inline KERNEL_FUNC void operator()(const sycl::nd_item<3> &ei,
                                     const arguments_t &args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context for flash mha loops
    ctx.init_context(ei, args);
    // preload Qi to local memory
    preload_Qi(args);
    // initialize matAccOi for accumulate the output
    matAccOi_t matAccOi(0);

    uint32_t startF = ei.get_group(1) * kBr;
    uint32_t endF = std::min(startF + kBr, args.uF);

    // iterate through the keys
    for (uint32_t startT = 0; startT < args.uT; startT += kBc) {
      if constexpr (kIsCausal) {
        if (startT >= endF)
          break;
      }
      // update context for current loop
      ctx.update_context(ei, args, startT);
      // compute Sij
      matAccSij_t matAccSij(0);
      gemm_Sij(&matAccSij, args);
      // apply mask
      apply_mask(&matAccSij, args, startF, startT);
      // softmax
      softmax_fwd(&matAccSij, &matAccOi, args);
      // compute Oi
      gemm_Oi(&matAccOi, args, startT);
    }

    // Store output to global
    store_Oi(matAccOi, args);
    store_for_backward(args);
  }
}; // fmha_forward_t

template <typename fmha_policy, typename T, bool kUseBias, bool kIsCausal,
          bool kIsTraining>
class FmhaForwardKernel;

// The launcher of fmha forward kernel
template <typename fmha_policy, typename T, bool kUseBias = false,
          bool kIsCausal = false, bool kIsTraining = false>
sycl::event fmha_forward_impl(sycl::queue &q,
                              void *_out, void *_dropout_mask, void *_bias,
                              void *_m, void *_l, uint32_t num_batches,
                              uint32_t num_heads, uint32_t head_size,
                              uint32_t num_queries, uint32_t num_keys,
                              uint64_t seed = 0, uint64_t offset = 123) {

  Shape shape(num_batches, num_heads, num_queries, num_keys, head_size);

  constexpr bool use_mask = false;
  constexpr bool use_dropout = false;
  float dropout_prob = 0.0f;
  if constexpr (use_dropout)
    dropout_prob = 0.5f;
  const float scale = 1 / (1 - dropout_prob);
  const float head_scale = sycl::rsqrt(float(head_size));

  uint32_t size_query = shape.get_query_size();
  uint32_t size_key = shape.get_key_size();
  uint32_t size_score = shape.get_score_size();
  uint32_t size_attn_mask = shape.get_attn_mask_size();
  uint32_t size_ml = shape.get_ml_size();

  // forward
  T *query = sycl::malloc_shared<T>(size_query, q);
  T *key = sycl::malloc_shared<T>(size_key, q);
  T *value = sycl::malloc_shared<T>(size_key, q);

  // T *bias = sycl::malloc_shared<T>(size_attn_mask, q);
  T *bias = static_cast<T *>(_bias);
  // uint8_t *dropout_mask = sycl::malloc_shared<uint8_t>(size_score, q);
  uint8_t *dropout_mask = static_cast<uint8_t *>(_dropout_mask);
  // T *out = sycl::malloc_shared<T>(size_query, q);
  T *out = static_cast<T *>(_out);
  // float *m = sycl::malloc_shared<float>(size_ml, q);
  float *m = static_cast<float *>(_m);
  // float *l = sycl::malloc_shared<float>(size_ml, q);
  float *l = static_cast<float *>(_l);

  // fmha forward kernel
  using fmha_forward_op_t =
      fmha_forward_t<fmha_policy, T, kUseBias, kIsCausal, kIsTraining>;

  sycl::nd_range<3> NdRange =
      fmha_forward_op_t::get_nd_range(num_batches * num_heads, num_queries);

  auto event = q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class FmhaForwardKernel<fmha_policy, T, kUseBias,
                                             kIsCausal, kIsTraining>>(
        NdRange, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          // exec item
          sycl::nd_item<3> ei(item);

          // init fmha forward op and arguments
          fmha_forward_op_t fmha_fwd_op;
          typename fmha_forward_op_t::arguments_t args(
              query, key, value, bias, dropout_prob, out, l, num_batches,
              num_heads, head_size, num_queries, num_keys, head_scale, seed,
              offset);

          // call the functor
          fmha_fwd_op(ei, args);
        });
  });
  // sycl::free(query, q);
  // sycl::free(key, q);
  // sycl::free(value, q);
  // sycl::free(bias, q);
  // sycl::free(dropout_mask, q);
  // sycl::free(out, q);
  return event;
}

} // namespace fmha

} // namespace gpu::xetla

#endif // TRITONBENCHMARK_FMHA_FWD_V5_H
