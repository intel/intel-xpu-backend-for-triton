=============================
Profiling kernels compilation
=============================

If some kernel takes to much time to compile, the first step to start the profiling with
is to run the program with the following environment variables:

.. code-block:: bash

    export TRITON_ALWAYS_COMPILE=1
    export MLIR_ENABLE_TIMING=1
    export LLVM_ENABLE_TIMING=1
    python -m pytest python/test/unit/intel/test_mxfp_matmul.py::test_mxfp_matmul[True-True-float4-float4-True-True-1-128-128-128-1024-512-512] --device=xpu -s

For the variables description, see the README.md file in the root directory of the project.
The output will contain the statistic for each phase of the compilation and for MLIR passes.
Here is an example:

.. code-block:: text

    ===-------------------------------------------------------------------------===
                            ... Execution time report ...
    ===-------------------------------------------------------------------------===
    Total Execution Time: 483.1849 seconds

    ----User Time----  ----Wall Time----  ----Name----
        0.0012 (  0.0%)    0.0012 (  0.0%)  SCFToControlFlowPass
        0.0015 (  0.0%)    0.0015 (  0.0%)  ConvertIndexToLLVMPass
        0.0516 (  0.0%)    0.0516 (  0.0%)  IntelAllocateSharedMemory
        0.0011 (  0.0%)    0.0011 (  0.0%)  TritonGPUGlobalScratchAllocationPass
        0.5947 (  0.1%)    0.5947 (  0.1%)  ConvertTritonIntelGPUToLLVM
        0.1232 (  0.0%)    0.1232 (  0.0%)  ConvertTritonGENToLLVM
        0.3430 (  0.1%)    0.3430 (  0.1%)  TritonIntelGPURewriteStackPtr
        15.3707 (  3.1%)   15.3707 (  3.2%)  Canonicalizer
        0.1650 (  0.0%)    0.1650 (  0.0%)  CSE
        0.0000 (  0.0%)    0.0000 (  0.0%)    (A) DominanceInfo
        0.0775 (  0.0%)    0.0775 (  0.0%)  ArithToLLVMConversionPass
        0.1934 (  0.0%)    0.1934 (  0.0%)  Canonicalizer
        0.1046 (  0.0%)    0.1046 (  0.0%)  CSE
        0.0000 (  0.0%)    0.0000 (  0.0%)    (A) DominanceInfo
        0.1470 (  0.0%)    0.1470 (  0.0%)  SymbolDCE
        0.0876 (  0.0%)    0.0876 (  0.0%)  LLVMDIScope
        483.1849 ( 96.6%)  465.9228 ( 96.4%)  Rest
        500.4471 (100.0%)  483.1849 (100.0%)  Total
            49 function calls in 465.865 seconds

What we can see here is that the most time-consuming part of the compilation is something,
which is called "Rest" and we don't know what it is. For further investigation, we can
use the Python's `cProfiler`. To profile not the entire program, but only individual functions,
we can use the following decorator:

.. code-block:: python

    def profile(func):
        import cProfile
        import pstats
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Profiling {func.__qualname__}")
            pr = cProfile.Profile()
            pr.enable()
            try:
                return func(*args, **kwargs)
            finally:
                pr.disable()
                ps = pstats.Stats(pr).sort_stats('cumulative')
                ps.print_stats(20)
        return wrapper

After decorating the `make_llir` function:

.. code-block:: python

    @profile
    def make_llir(...):
        ...

and running the program again, we will get the following results:

.. code-block:: text

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1  465.552  465.552  465.552  465.552 {built-in method triton._C.libtriton.intel.optimize_module}
        1    0.231    0.231    0.231    0.231 {built-in method triton._C.libtriton.llvm.to_module}
        ...

Now we see that the most time is spent in the `optimize_module` function, which is a native function,
written in C++. To profile this function, we will use `gperftools`. On an Ubuntu system, we need
the following packages to be installed:

.. code-block:: bash

    sudo apt install google-perftools libgoogle-perftools-dev

We also need to link the Triton library with the `profiler` by editing following line in the
`CMakeLists.txt` file:

.. code-block:: cmake

    --target_link_libraries(TritonXPU PRIVATE Python3::Module pybind11::headers)
    ++target_link_libraries(TritonXPU PRIVATE Python3::Module pybind11::headers profiler)

The `optimize_module` function is a python binding implemented in `third_party/intel/triton_xpu.cc`.
To profile it, we need to add the following lines to the code:

.. code-block:: cpp

    ++#include <gperftools/profiler.h>

    ...

      "optimize_module",
      [](llvm::Module *mod, const llvm::OptimizationLevel &opt,
         std::string arch, std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion) {
         ++ProfilerStart("optimize_module.prof");
         ...
         ++ProfilerStop();

In the code above, the `"optimize_module.prof"` string is the file path, where the profiling results will be saved.
After running the program, we will get the `optimize_module.prof` binary file, which can be visualized using the
`pprof-symbolize` tool:

.. code-block:: bash

    pprof-symbolize --text /usr/bin/python3 optimize_module.prof

    Total: 42300 samples
   13378  31.6%  31.6%    14589  34.5% __default_morecore@GLIBC_2.2.5
    7939  18.8%  50.4%     7939  18.8% llvm::APInt::countTrailingOnesSlowCase
    5810  13.7%  64.1%    11998  28.4% malloc@@GLIBC_2.2.5
    5007  11.8%  76.0%     5007  11.8% llvm::APInt::orAssignSlowCase
    3996   9.4%  85.4%    17814  42.1% llvm::APInt::initSlowCase
    2237   5.3%  90.7%    42125  99.6% findDemandedEltsByAllUsers
    1211   2.9%  93.6%     1211   2.9% timer_settime@GLIBC_2.2.5
    1153   2.7%  96.3%     1153   2.7% __nss_database_lookup@GLIBC_2.2.5
     538   1.3%  97.6%      538   1.3% _fini
     402   1.0%  98.5%     8803  20.8% free@@GLIBC_2.2.5
     160   0.4%  98.9%    12309  29.1% operator new@@GLIBCXX_3.4
     154   0.4%  99.3%      154   0.4% std::__once_callable@@GLIBCXX_3.4.11
     141   0.3%  99.6%      141   0.3% operator delete@@GLIBCXX_3.4
      14   0.0%  99.6%       14   0.0% llvm::BasicBlock::renumberInstructions
      12   0.0%  99.7%    42176  99.7% llvm::InstCombinerImpl::run
      11   0.0%  99.7%       12   0.0% llvm::SymbolTableListTraits::addNodeToList
      11   0.0%  99.7%       22   0.1% passingValueIsAlwaysUndefined
       7   0.0%  99.7%        7   0.0% llvm::AttributeList::hasFnAttr
       5   0.0%  99.7%        8   0.0% llvm::Instruction::mayThrow
       3   0.0%  99.7%        3   0.0% llvm::User::isDroppable
       3   0.0%  99.7%        3   0.0% llvm::Value::getName
       3   0.0%  99.8%        3   0.0% llvm::Value::setValueName
       3   0.0%  99.8%        9   0.0% llvm::isa@64af770
       2   0.0%  99.8%        2   0.0% isBlockInLCSSAForm
       2   0.0%  99.8%        5   0.0% llvm::CallBase::hasFnAttrOnCalledFunction
       2   0.0%  99.8%        7   0.0% llvm::InstCombinerImpl::visitCallBase
       2   0.0%  99.8%    42134  99.6% llvm::InstCombinerImpl::visitExtractElementInst

The 5-th column shows the percentage of the samples, counted for each function.
From the above report, we can see, that the most time is spent in the
`InstCombinerImpl::run/visitExtractElementInst` functions. Most probably, the optimizer spent
so much time, analysing the `extractelement` operations. If we look at the Triton cache
`~/.triton/cache`, we will find the `kernel.llir` file, which contains a huge amount
of `extractelement` operations, that confirms our assumption.
