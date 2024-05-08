/*
 * Copyright (c) 2020, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "kernel_func.hpp"

using namespace gpu::xetla;

_GENX_MAIN_ void bgemm(data_type_a *A [[type("svmptr_t")]],
                       data_type_b *B [[type("svmptr_t")]],
                       data_type_c *C [[type("svmptr_t")]], int matrix_m,
                       int matrix_n, int matrix_k, int lda, int ldb, int ldc,
                       data_type_acc *Acc [[type("svmptr_t")]],
                       uint32_t *Cnt [[type("svmptr_t")]]) {

  sycl::nd_item<3> item;
  using bgemm_functor =
      bgemm_test_func<data_type_a, data_type_b, data_type_c, data_type_acc,
                      wg_tile_m_d, wg_tile_n_d, sg_tile_m_d, sg_tile_n_d,
                      sg_tile_k_d, MEM_LAYOUT_A, MEM_LAYOUT_B,
                      global_kslicing_d, local_kslicing_d>;

  constexpr uint32_t barrier_count = bgemm_functor::barrier_count;
  constexpr uint32_t slm_size = bgemm_functor::slm_size;
  if constexpr (barrier_count != 0) {
    cm_nbarrier_init(barrier_count);
  }
  if constexpr (slm_size != 0) {
    cm_slm_init(slm_size);
  }

  bgemm_functor::run(item, A, B, C, matrix_m, matrix_n, matrix_k, lda, ldb, ldc,
                     Acc, Cnt);
}
