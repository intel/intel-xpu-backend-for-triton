#ifndef TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H
#define TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H

#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include <string>

namespace mlir::triton::gpu::intel {

/// The device and options the pipeliner was invoked for; a device without
/// prefetch cost model constants or a clock keeps every dot-operand prefetch.
struct PipelineTarget {
  std::string deviceArch;
  unsigned coreClockKhz = 0;
  unsigned memClockMhz = 0;
  bool useBarrier = false;
  bool tileBarrier = false;
};

bool preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                  const PipelineTarget &target,
                                  mlir::scf::PipeliningOption &options);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_SCHEDULE_H
