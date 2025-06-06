/*******************************************************************************
 * Copyright 2020-2025 Intel Corporation
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

/// @example softmax.cpp
/// > Annotated version: @ref softmax_example_cpp
///
/// @page softmax_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Softmax](@ref dev_guide_softmax) primitive in forward training propagation
/// mode.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Softmax along axis 1 (C) for 2D tensors.
///
/// @page softmax_example_cpp Softmax Primitive Example
/// @copydetails softmax_example_cpp_short
///
/// @include softmax.cpp
#ifndef TRITONBENCHMARK_ONEDNN_SOFTMAX_H
#define TRITONBENCHMARK_ONEDNN_SOFTMAX_H

#include <algorithm>
#include <iostream>

#include "../examples/example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

dnnl::stream softmax_example(const int M, const int N, const int axis,
                             void *input, void *output, dnnl::engine &engine,
                             dnnl::stream engine_stream) {

  try {
    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, N};

    // Create memory descriptor and memory object.
    auto mem_desc =
        memory::desc(src_dims, memory::data_type::bf16, memory::format_tag::nc);
    auto src_mem = memory(mem_desc, engine, input);
    auto dst_mem = memory(mem_desc, engine, output);

    // Create primitive descriptor.
    auto softmax_pd = softmax_forward::primitive_desc(
        engine, prop_kind::forward_training, algorithm::softmax_accurate,
        mem_desc, mem_desc, axis);

    // Create the primitive.
    auto softmax_prim = softmax_forward(softmax_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    return engine_stream;
  } catch (dnnl::error &e) {
    std::cout << "oneDNN error caught: " << std::endl
              << "\tStatus: " << dnnl_status2str(e.status) << std::endl
              << "\tMessage: " << e.what() << std::endl;
  }
}
#endif // TRITONBENCHMARK_ONEDNN_SOFTMAX_H
