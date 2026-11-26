// Registration of the XPU profiler with Proton when it is built as a library of
// its own (PROTON_XPU_EXTERNAL), e.g. next to a Triton core that does not
// contain it. libproton compiles the XPU profiler in and does not use this
// file.
//
// On Linux, loading the library registers the profiler, device and runtime
// through proton::registerBackend(), with xpu::kDeviceType, which is
// DeviceType::EXTERNAL here. The two values the profiler needs from Python, the
// SYCL queue and the path of the helper library, come from the active Triton
// driver when a session creates the profiler, where libproton takes them from
// libproton.start() (Session.cpp).
#ifdef PROTON_XPU_EXTERNAL

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>
#include <utility>

#include "Backend/Backend.h"
#include "Driver/GPU/XpuApi.h"
#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Runtime/XpuRuntime.h"

namespace proton {

namespace {

namespace nb = nanobind;

// Runs under Proton's session lock with the GIL held, so it must not call
// Proton. If the Python call hands the GIL to a thread that then waits for the
// lock in a Proton scope or op, the start hangs. Fixing that needs Proton to
// call registered factories outside the lock.
// A Python error leaves as nb::python_error, which libproton's nanobind
// bindings raise again from proton.start().
void prepareSession() {
  nb::gil_scoped_acquire gil;
  auto [queue, path] = nb::cast<std::pair<uintptr_t, std::string>>(
      nb::module_::import_("triton.runtime")
          .attr("driver")
          .attr("active")
          .attr("prepare_proton_session")());
  XpuRuntime::instance().setSyclQueue(reinterpret_cast<void *>(queue));
  xpu::PROTON_UTILS = path;
}

BackendRegistration makeRegistration() {
  return {
      ProfilerRegistration{"xpupti", "xpu",
                           []() -> Profiler * {
                             prepareSession();
                             return &XpuptiProfiler::instance();
                           }},
      DeviceRegistration{"XPU", xpu::kDeviceType,
                         [](uint64_t index) { return xpu::getDevice(index); }},
      RuntimeRegistration{
          "XPU", []() -> Runtime * { return &XpuRuntime::instance(); }},
  };
}

} // namespace

} // namespace proton

extern "C" {

// Idempotent; also called from Python where there is no load-time constructor.
#if defined(_WIN32)
__declspec(dllexport)
#endif
int proton_xpu_register() {
  static const int registered = [] {
    ::proton::registerBackend(::proton::makeRegistration());
    return 1;
  }();
  return registered;
}

#if !defined(_WIN32)
__attribute__((constructor)) static void protonXpuAutoRegister() {
  proton_xpu_register();
}
#endif
}

#endif // PROTON_XPU_EXTERNAL
