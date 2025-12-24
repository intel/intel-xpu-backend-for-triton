add_proton_library(ProtonRuntime
  CudaRuntime.cpp
  HipRuntime.cpp
  # FIXME: implement Xpu Runtime API
  # XpuRuntime.cpp
)
