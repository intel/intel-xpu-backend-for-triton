/*
 * Minimal stub for Triton-distributed: this header is included by
 * ConvertAMDDistributedToLLVM.cpp. The AMD dialect (Dialect.h) provides
 * InstructionSchedHint; no additional declarations are needed here.
 */
#ifndef TRITON_AMD_SCHED_INSTRUCTIONS_H
#define TRITON_AMD_SCHED_INSTRUCTIONS_H

// InstructionSchedHint and related types come from the AMD dialect.
// ConvertAMDDistributedToLLVM.cpp already includes the dialect;
// this stub exists so the #include "third_party/amd/lib/.../SchedInstructions.h"
// resolves when building with TRITON_ENABLE_AMD=ON.

#endif // TRITON_AMD_SCHED_INSTRUCTIONS_H
