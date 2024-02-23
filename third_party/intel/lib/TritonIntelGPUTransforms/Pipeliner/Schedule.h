#ifndef TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "mlir/Dialect/SCF/Transforms/Transforms.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

bool preProcessLoopAndGetScheduleIntel(scf::ForOp &forOp, int numStages,
                                       mlir::scf::PipeliningOption &options);

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
