#ifndef TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H
#define TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H

#include "mlir/Dialect/SCF/Transforms/Transforms.h"

namespace mlir::triton::gpu::intel {

bool preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                  mlir::scf::PipeliningOption &options);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H
