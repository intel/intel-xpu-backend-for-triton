/*========================== begin_copyright_notice ============================

Copyright (C) 2023 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/
#pragma once

//#include "common/LLVMWarningsPush.hpp"
#include "llvm/IR/Intrinsics.h"
//#include "common/LLVMWarningsPop.hpp"

#include <stdint.h>

namespace llvm
{

namespace GenISAIntrinsic {

enum ID : uint32_t
{
    no_intrinsic = llvm::Intrinsic::num_intrinsics,
    GenISA_2fto2bf,
    GenISA_assume_uniform,
    GenISA_bftof,
    GenISA_CatchAllDebugLine,
    GenISA_DCL_DSCntrlPtInputVec,
    GenISA_DCL_DSInputTessFactor,
    GenISA_DCL_DSPatchConstInputVec,
    GenISA_DCL_GSinputVec,
    GenISA_DCL_GSsystemValue,
    GenISA_DCL_HSControlPointID,
    GenISA_DCL_HSOutputCntrlPtInputVec,
    GenISA_DCL_HSPatchConstInputVec,
    GenISA_DCL_HSinputVec,
    GenISA_DCL_ShaderInputVec,
    GenISA_DCL_SystemValue,
    GenISA_DCL_input,
    GenISA_DCL_inputVec,
    GenISA_dpas,
    GenISA_EmitHitAttributes,
    GenISA_EndPrimitive,
    GenISA_ftobf,
    GenISA_GetBufferPtr,
    GenISA_GetImplicitBufferPtr,
    GenISA_GetLocalIdBufferPtr,
    GenISA_GetPixelMask,
    GenISA_GradientX,
    GenISA_GradientXfine,
    GenISA_GradientY,
    GenISA_GradientYfine,
    GenISA_GsCutControlHeader,
    GenISA_GsStreamHeader,
    GenISA_HSURBPatchHeaderRead,
    GenISA_IEEE_Divide,
    GenISA_IEEE_Sqrt,
    GenISA_InitDiscardMask,
    GenISA_InnerScalarTessFactors,
    GenISA_Interpolant,
    GenISA_Interpolate,
    GenISA_Interpolate2,
    GenISA_IsHelperInvocation,
    GenISA_MediaBlockRead,
    GenISA_MediaBlockRectangleRead,
    GenISA_MediaBlockWrite,
    GenISA_OUTPUT,
    GenISA_OUTPUTGS,
    GenISA_OuterScalarTessFactors,
    GenISA_OutputTessControlPoint,
    GenISA_PHASE_INPUT,
    GenISA_PHASE_INPUTVEC,
    GenISA_PHASE_OUTPUT,
    GenISA_PHASE_OUTPUTVEC,
    GenISA_PatchConstantOutput,
    GenISA_PixelPositionX,
    GenISA_PixelPositionY,
    GenISA_PullCentroidBarys,
    GenISA_PullSampleIndexBarys,
    GenISA_PullSnappedBarys,
    GenISA_QuadPrefix,
    GenISA_ROUNDNE,
    GenISA_RTDualBlendSource,
    GenISA_RTWrite,
    GenISA_ReadFromReservedArgSpace,
    GenISA_RenderTargetRead,
    GenISA_RenderTargetReadSampleFreq,
    GenISA_RuntimeValue,
    GenISA_SampleOffsetX,
    GenISA_SampleOffsetY,
    GenISA_SaveInReservedArgSpace,
    GenISA_SetStackCallsBaseAddress,
    GenISA_SetImplicitBufferPtr,
    GenISA_SetDebugReg,
    GenISA_SetLocalIdBufferPtr,
    GenISA_SetStream,
    GenISA_StackAlloca,
    GenISA_VLAStackAlloca,
    GenISA_UnmaskedRegionBegin,
    GenISA_UnmaskedRegionEnd,
    GenISA_URBRead,
    GenISA_URBReadOutput,
    GenISA_URBWrite,
    GenISA_UpdateDiscardMask,
    GenISA_WaveAll,
    GenISA_WaveBallot,
    GenISA_WaveClustered,
    GenISA_WaveInverseBallot,
    GenISA_WavePrefix,
    GenISA_WaveShuffleIndex,
    GenISA_WaveBroadcast,
    GenISA_WorkGroupAny,
    GenISA_add_pair,
    GenISA_add_rtz,
    GenISA_atomiccounterinc,
    GenISA_atomiccounterpredec,
    GenISA_bfi,
    GenISA_bfrev,
    GenISA_broadcastMessagePhase,
    GenISA_broadcastMessagePhaseV,
    GenISA_cmpSADs,
    GenISA_cmpxchgatomicstructured,
    GenISA_createMessagePhases,
    GenISA_createMessagePhasesNoInit,
    GenISA_createMessagePhasesNoInitV,
    GenISA_createMessagePhasesV,
    GenISA_cycleCounter,
    GenISA_discard,
    GenISA_dp4a_ss,
    GenISA_dp4a_su,
    GenISA_dp4a_us,
    GenISA_dp4a_uu,
    GenISA_dummyInst,
    GenISA_dummyInstID,
    GenISA_launder,
    GenISA_dwordatomicstructured,
    GenISA_eu_id,
    GenISA_eu_thread_id,
    GenISA_eu_thread_pause,
    GenISA_evaluateSampler,
    GenISA_extractMVAndSAD,
    GenISA_f32tof16_rtz,
    GenISA_fcmpxchgatomicraw,
    GenISA_fcmpxchgatomicrawA64,
    GenISA_fcmpxchgatomicstructured,
    GenISA_firstbitHi,
    GenISA_firstbitLo,
    GenISA_firstbitShi,
    GenISA_floatatomicraw,
    GenISA_floatatomicrawA64,
    GenISA_floatatomicstructured,
    GenISA_flushsampler,
    GenISA_fma_rtz,
    GenISA_fma_rtp,
    GenISA_fma_rtn,
    GenISA_fsat,
    GenISA_usat,
    GenISA_isat,
    GenISA_ftof_rte,
    GenISA_ftof_rtn,
    GenISA_ftof_rtp,
    GenISA_ftof_rtz,
    GenISA_ftoi_rte,
    GenISA_ftoi_rtn,
    GenISA_ftoi_rtp,
    GenISA_ftoui_rte,
    GenISA_ftoui_rtn,
    GenISA_ftoui_rtp,
    GenISA_sampleMlodptr,
    GenISA_sampleCMlodptr,
    GenISA_sampleBCMlodptr,
    GenISA_sampleDCMlodptr,
    GenISA_samplePOptr,
    GenISA_samplePOBptr,
    GenISA_samplePOLptr,
    GenISA_samplePOCptr,
    GenISA_samplePODptr,
    GenISA_gather4Iptr,
    GenISA_gather4Bptr,
    GenISA_gather4Lptr,
    GenISA_samplePOLCptr,
    GenISA_gather4ICptr,
    GenISA_gather4LCptr,
    GenISA_gather4POPackedptr,
    GenISA_gather4POPackedLptr,
    GenISA_gather4POPackedBptr,
    GenISA_gather4POPackedIptr,
    GenISA_gather4POPackedICptr,
    GenISA_gather4POPackedLCptr,
    GenISA_gather4POPackedCptr,
    GenISA_gather4IPOptr,
    GenISA_gather4BPOptr,
    GenISA_gather4LPOptr,
    GenISA_gather4ICPOptr,
    GenISA_gather4LCPOptr,
    GenISA_gather4Cptr,
    GenISA_gather4POCptr,
    GenISA_gather4POptr,
    GenISA_gather4ptr,
    GenISA_getMessagePhase,
    GenISA_getMessagePhaseV,
    GenISA_getMessagePhaseX,
    GenISA_getMessagePhaseXV,
    GenISA_getR0,
    GenISA_getPayloadHeader,
    GenISA_getWorkDim,
    GenISA_getNumWorkGroups,
    GenISA_getGlobalSize,
    GenISA_getLocalSize,
    GenISA_getEnqueuedLocalSize,
    GenISA_getLocalID_X,
    GenISA_getLocalID_Y,
    GenISA_getLocalID_Z,
    GenISA_getPrivateBase,
    GenISA_getPrintfBuffer,
    GenISA_getStageInGridOrigin,
    GenISA_getStageInGridSize,
    GenISA_getSyncBuffer,
    GenISA_getRtGlobalBufferPtr,
    GenISA_getStackPointer,
    GenISA_getStackSizePerThread,
    GenISA_getAssertBufferPtr,
    GenISA_getSR0,
    GenISA_getSR0_0,
    GenISA_globalSync,
    GenISA_hw_thread_id,
    GenISA_hw_thread_id_alloca,
    GenISA_ibfe,
    GenISA_icmpxchgatomicraw,
    GenISA_icmpxchgatomicrawA64,
    GenISA_icmpxchgatomictyped,
    GenISA_fcmpxchgatomictyped,
    GenISA_imulH,
    GenISA_intatomicraw,
    GenISA_intatomicrawA64,
    GenISA_intatomictyped,
    GenISA_floatatomictyped,
    GenISA_is_uniform,
    GenISA_itof_rtn,
    GenISA_itof_rtp,
    GenISA_itof_rtz,
    GenISA_ldmcsptr,
    GenISA_ldmsptr,
    GenISA_ldmsptr16bit,
    GenISA_ldptr,
    GenISA_ldlptr,
    GenISA_ldraw_indexed,
    GenISA_ldrawvector_indexed,
    GenISA_ldstructured,
    GenISA_lodptr,
    GenISA_memoryfence,
    GenISA_mov_identity,
    GenISA_movcr,
    GenISA_movflag,
    GenISA_software_exception,
    GenISA_mul_pair,
    GenISA_mul_rtz,
    GenISA_pair_to_ptr,
    GenISA_patchInstanceId,
    GenISA_ptr_to_pair,
    GenISA_readsurfacetypeandformat,
    GenISA_resinfoptr,
    GenISA_rsq,
    GenISA_sampleBCptr,
    GenISA_sampleBptr,
    GenISA_sampleCptr,
    GenISA_sampleDCptr,
    GenISA_sampleDptr,
    GenISA_sampleKillPix,
    GenISA_sampleLCptr,
    GenISA_sampleLptr,
    GenISA_sampleinfoptr,
    GenISA_sampleptr,
    GenISA_setMessagePhase,
    GenISA_setMessagePhaseV,
    GenISA_setMessagePhaseX,
    GenISA_setMessagePhaseXV,
    GenISA_setMessagePhaseX_legacy,
    GenISA_setMessagePhase_legacy,
    GenISA_simdBlockRead,
    GenISA_simdBlockReadBindless,
    GenISA_simdBlockWrite,
    GenISA_simdBlockWriteBindless,
    GenISA_simdGetMessagePhase,
    GenISA_simdGetMessagePhaseV,
    GenISA_simdLaneId,
    GenISA_simdLaneIdReplicate,
    GenISA_simdMediaBlockRead,
    GenISA_simdMediaBlockWrite,
    GenISA_simdMediaRegionCopy,
    GenISA_simdSetMessagePhase,
    GenISA_simdSetMessagePhaseV,
    GenISA_simdShuffleDown,
    GenISA_simdShuffleXor,
    GenISA_simdSize,
    GenISA_slice_id,
    GenISA_source_value,
    GenISA_storeraw_indexed,
    GenISA_storerawvector_indexed,
    GenISA_storestructured1,
    GenISA_storestructured2,
    GenISA_storestructured3,
    GenISA_storestructured4,
    GenISA_sub_group_dpas,
    GenISA_sub_pair,
    GenISA_subslice_id,
    GenISA_logical_subslice_id,
    GenISA_dual_subslice_id,
    GenISA_threadgroupbarrier,
    GenISA_threadgroupbarrier_signal,
    GenISA_threadgroupbarrier_wait,
    GenISA_typedmemoryfence,
    GenISA_typedread,
    GenISA_typedwrite,
    GenISA_uaddc,
    GenISA_uavSerializeAll,
    GenISA_uavSerializeOnResID,
    GenISA_ubfe,
    GenISA_uitof_rtn,
    GenISA_uitof_rtp,
    GenISA_uitof_rtz,
    GenISA_umulH,
    GenISA_usubb,
    GenISA_vaBoolCentroid,
    GenISA_vaBoolSum,
    GenISA_vaCentroid,
    GenISA_vaConvolve,
    GenISA_vaConvolveGRF_16x1,
    GenISA_vaConvolveGRF_16x4,
    GenISA_vaDilate,
    GenISA_vaErode,
    GenISA_vaMinMax,
    GenISA_vaMinMaxFilter,
    GenISA_vectorUniform,
    GenISA_vmeSendFBR,
    GenISA_vmeSendFBR2,
    GenISA_vmeSendIME,
    GenISA_vmeSendIME2,
    GenISA_vmeSendSIC,
    GenISA_vmeSendSIC2,
    GenISA_wavebarrier,
    GenISA_frc,
    GenISA_staticConstantPatchValue,
    GenISA_HDCCCSFastClear,
    GenISA_LSC2DBlockRead,
    GenISA_LSC2DBlockWrite,
    GenISA_LSC2DBlockPrefetch,
    GenISA_LSCAtomicFP32,
    GenISA_LSCAtomicFP64,
    GenISA_LSCAtomicInts,
    GenISA_LSCFence,
    GenISA_LSCLoad,
    GenISA_LSCLoadCmask,
    GenISA_LSCLoadBlock,
    GenISA_LSCLoadStatus,
    GenISA_LSCPrefetch,
    GenISA_LSCStore,
    GenISA_LSCStoreCmask,
    GenISA_LSCStoreBlock,
    GenISA_bf8tohf,
    GenISA_tf32tof,
    GenISA_HDCuncompressedwrite,
    GenISA_systemmemoryfence,
    GenISA_urbfence,
    GenISA_threadgroupnamedbarriers_signal,
    GenISA_threadgroupnamedbarriers_wait,
    GenISA_hftobf8,
    GenISA_ftotf32,
    GenISA_srnd_hftobf8,
    GenISA_srnd_ftohf,
    GenISA_OutputMeshPrimitiveData,
    GenISA_OutputMeshPrimitiveDataInput,
    GenISA_OutputMeshSivDataInput,
    GenISA_OutputMeshVertexData,
    GenISA_OutputMeshVertexDataInput,
    GenISA_OutputTaskData,
    GenISA_OutputTaskDataInput,
    GenISA_AcceptHitAndEndSearchHL,
    GenISA_AllocaNumber,
    GenISA_AllocateRayQuery,
    GenISA_AsyncStackID,
    GenISA_AsyncStackPtr,
    GenISA_SyncStackPtr,
    GenISA_BindlessThreadDispatch,
    GenISA_CallShaderHL,
    GenISA_DispatchDimensions,
    GenISA_DispatchRayIndex,
    GenISA_FillValue,
    GenISA_GetShaderRecordPtr,
    GenISA_GlobalBufferPointer,
    GenISA_GlobalRootSignatureValue,
    GenISA_HitKind,
    GenISA_IgnoreHitHL,
    GenISA_InlinedData,
    GenISA_LocalBufferPointer,
    GenISA_LocalRootSignatureValue,
    GenISA_PayloadPtr,
    GenISA_PreemptionEnable,
    GenISA_PreemptionDisable,
    GenISA_RayQueryCheck,
    GenISA_RayQueryRelease,
    GenISA_ContinuationSignpost,
    GenISA_RTStatefulBTIAndOffset,
    GenISA_RayInfo,
    GenISA_RayTCurrent,
    GenISA_ReportHitHL,
    GenISA_TileXOffset,
    GenISA_TileYOffset,
    GenISA_SpillValue,
    GenISA_StackIDRelease,
    GenISA_StackSize,
    GenISA_SWHotZonePtr,
    GenISA_SWStackPtr,
    GenISA_TraceRayAsync,
    GenISA_TraceRaySync,
    GenISA_TraceRaySyncProceed,
    GenISA_ShadowMemoryToSyncStack,
    GenISA_SyncStackToShadowMemory,
    GenISA_ReadTraceRaySync,
    GenISA_TraceRayAsyncHL,
    GenISA_TraceRayInlineAbort,
    GenISA_TraceRayInlineCandidateType,
    GenISA_TraceRayInlineCommitNonOpaqueTriangleHit,
    GenISA_TraceRayInlineCommitProceduralPrimitiveHit,
    GenISA_TraceRayInlineCommittedStatus,
    GenISA_TraceRayInlineHL,
    GenISA_TraceRaySyncProceedHL,
    GenISA_TraceRayInlineRayInfo,
    GenISA_rt_swstack_offset,
    GenISA_FPBinaryOperator,
    GenISA_bitcastfromstruct,
    GenISA_bitcasttostruct,
    num_genisa_intrinsics
};

} // namespace GenISAIntrinsic

} // namespace llvm
