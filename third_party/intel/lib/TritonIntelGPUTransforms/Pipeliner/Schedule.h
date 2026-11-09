#ifndef TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H
#define TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H

#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include <cstdint>
#include <optional>

namespace mlir::triton::gpu::intel {

bool preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                  mlir::scf::PipeliningOption &options,
                                  std::optional<int64_t> &prefetchBytes);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H
