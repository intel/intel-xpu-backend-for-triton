#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace mlir::triton::intel {

/// Annotate 2D block I/O SPIR-V builtins with precise memory attributes.
///
/// The block load/store/prefetch builtins have mangled names that include the
/// full parameter signature (e.g.,
/// `_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv`), so we must
/// iterate over all functions and match by prefix.  Adding `argmemonly` (for
/// loads/stores) and `inaccessiblemem_or_argmemonly` (for prefetches) allows
/// LLVM to reason about aliasing and enables better instruction scheduling.
static void annotateBlockIOBuiltins(llvm::Module &mod) {
  // Prefixes for the three categories of 2D block I/O builtins.
  // Loads read from global memory through a pointer argument.
  static constexpr llvm::StringLiteral loadPrefixes[] = {
      "_Z32__spirv_Subgroup2DBlockLoadINTEL",
      "_Z41__spirv_Subgroup2DBlockLoadTransposeINTEL",
      "_Z41__spirv_Subgroup2DBlockLoadTransformINTEL",
  };

  // Stores write to global memory through a pointer argument.
  static constexpr llvm::StringLiteral storePrefix =
      "_Z33__spirv_Subgroup2DBlockStoreINTEL";

  // Prefetches only initiate a cache fill; they are effectively
  // inaccessible-memory-or-argmem-only (the data goes to cache, not to a
  // register visible to LLVM).
  static constexpr llvm::StringLiteral prefetchPrefix =
      "_Z36__spirv_Subgroup2DBlockPrefetchINTEL";

  for (llvm::Function &fn : mod) {
    if (!fn.isDeclaration())
      continue;

    llvm::StringRef name = fn.getName();

    // Check loads: read argmem only.
    for (const auto &prefix : loadPrefixes) {
      if (name.starts_with(prefix)) {
        fn.setOnlyAccessesArgMemory();
        fn.setDoesNotThrow();
        fn.setWillReturn();
        goto next;
      }
    }

    // Check store: write argmem only.
    if (name.starts_with(storePrefix)) {
      fn.setOnlyAccessesArgMemory();
      fn.setDoesNotThrow();
      fn.setWillReturn();
      continue;
    }

    // Check prefetch: inaccessible-mem-or-argmem-only.
    if (name.starts_with(prefetchPrefix)) {
      fn.setOnlyAccessesInaccessibleMemOrArgMem();
      fn.setDoesNotThrow();
      fn.setWillReturn();
      continue;
    }

  next:;
  }
}

/// Mark SPIR-V work-item/sub-group ID builtins as memory-free.
///
/// Functions like get_sub_group_id(), get_group_id(), and
/// get_sub_group_local_invocation_id() return thread-constant values and never
/// access memory.  Without the `memory(none)` attribute LLVM's LICM pass
/// cannot prove they are loop-invariant and therefore does not hoist them out
/// of inner loops, leading to redundant calls in hot GEMM loops.
static void markPureBuiltins(llvm::Module &mod) {
  // These SPIR-V builtins are pure: they return a value that is constant for
  // the lifetime of the work-item/sub-group and have no memory side-effects.
  static const char *const pureBuiltins[] = {
      "_Z16get_sub_group_id",       "_Z12get_group_idj",
      "_Z22get_sub_group_local_id", "_Z33get_sub_group_local_invocation_id",
      "_Z13get_global_idj",         "_Z12get_local_idj",
      "_Z14get_local_sizej",        "_Z15get_global_sizej",
      "_Z14get_num_groupsj",
  };

  for (const char *name : pureBuiltins) {
    if (llvm::Function *fn = mod.getFunction(name)) {
      fn->setDoesNotAccessMemory();
      fn->setDoesNotThrow();
      fn->setWillReturn();
      fn->setMustProgress();
    }
  }
}

void postProcessLLVMIR(llvm::Module &mod) {
  // __devicelib_assert_fail must be a declaration so that
  // IGC can replace it with a runtime assert function.
  // If a 'fallback' implementation is defined in SYCL libarary, the
  // assertion does not work correctly.
  for (auto &f : mod) {
    if (f.getName().str() == "__devicelib_assert_fail") {
      assert(f.isDeclaration() &&
             "__devicelib_assert_fail must be a declaration!");
    }
  }

  markPureBuiltins(mod);
  // REGRESSION FIX: Disable annotateBlockIOBuiltins pending further
  // investigation. This function was introduced to annotate 2D block I/O SPIR-V
  // builtins with memory attributes, but testing reveals it causes a ~20%
  // performance regression. With it disabled, performance improves from 60% to
  // 87% of oneMKL baseline. Root cause: likely incorrect attribute semantics
  // with 2D block descriptors. annotateBlockIOBuiltins(mod);
}

} // namespace mlir::triton::intel
