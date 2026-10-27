// Registration of the XPU profiler with Proton when it is built as a library of
// its own (PROTON_XPU_EXTERNAL), e.g. next to a Triton core that does not
// contain it. libproton compiles the XPU profiler in and does not use this
// file.
//
// Loading the library registers the profiler through proton::registerBackend(),
// which gives it DeviceType::EXTERNAL. The two values the profiler needs from
// Python, the SYCL queue and the path of the helper library, come from the
// active Triton driver when a session creates the profiler, where libproton
// takes them from libproton.start() (Session.cpp).
#ifdef PROTON_XPU_EXTERNAL

#include <Python.h>

#include "Backend/Backend.h"
#include "Driver/GPU/XpuApi.h"
#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Runtime/XpuRuntime.h"

namespace proton {

namespace {

void prepareSession() {
  if (!Py_IsInitialized())
    return;
  PyGILState_STATE gil = PyGILState_Ensure();
  // triton.runtime.driver.active.prepare_proton_session() -> (queue, path)
  PyObject *result = nullptr;
  if (PyObject *runtime = PyImport_ImportModule("triton.runtime")) {
    if (PyObject *driver = PyObject_GetAttrString(runtime, "driver")) {
      if (PyObject *active = PyObject_GetAttrString(driver, "active")) {
        result = PyObject_CallMethod(active, "prepare_proton_session", nullptr);
        Py_DECREF(active);
      }
      Py_DECREF(driver);
    }
    Py_DECREF(runtime);
  }
  if (result && PyTuple_Check(result) && PyTuple_Size(result) == 2) {
    void *queue = PyLong_AsVoidPtr(PyTuple_GetItem(result, 0));
    const char *path = PyUnicode_AsUTF8(PyTuple_GetItem(result, 1));
    XpuRuntime::instance().setSyclQueue(queue);
    if (path)
      xpu::PROTON_UTILS = path;
  }
  Py_XDECREF(result);
  // A session start must not raise into an unrelated Python frame.
  if (PyErr_Occurred())
    PyErr_Print();
  PyGILState_Release(gil);
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
bool proton_xpu_register() {
  static const bool registered = [] {
    ::proton::registerBackend(::proton::makeRegistration());
    return true;
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
