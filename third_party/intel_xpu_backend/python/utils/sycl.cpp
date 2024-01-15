#include <cstdlib>
#include <iostream>
#include <level_zero/ze_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

namespace py = pybind11;
using namespace py::literals;

namespace {

bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

} // namespace

#define EXPECT_EQ(value1, value2)                                              \
  {                                                                            \
    auto result = (value2);                                                    \
    if ((value1) != (result)) {                                                \
      std::string err_log("L0 API error code: ");                              \
      std::stringstream ss;                                                    \
      ss << std::hex << result << std::endl;                                   \
      throw std::runtime_error(err_log + ss.str());                            \
    }                                                                          \
  }

#define EXPECT_TRUE(value1) EXPECT_EQ(true, value1)

ze_module_handle_t create_module(ze_context_handle_t context,
                                 ze_device_handle_t device,
                                 uint32_t *binary_ptr, size_t binary_size) {

  const char *build_flags = "";
  const ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;

  ze_module_desc_t module_description = {};
  module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  ze_module_constants_t module_constants = {};
  module_constants.numConstants = 0;
  module_constants.pConstantIds = 0;
  module_constants.pConstantValues = 0;

  module_description.pNext = nullptr;
  module_description.format = format;
  module_description.inputSize =
      static_cast<uint32_t>(binary_size * sizeof(uint32_t));
  module_description.pInputModule = (uint8_t *)binary_ptr;
  module_description.pBuildFlags = build_flags;
  module_description.pConstants = &module_constants;

  ze_module_build_log_handle_t buildlog;
  ze_module_handle_t module;
  auto context_initial = context;
  auto device_initial = device;
  auto error_no =
      zeModuleCreate(context, device, &module_description, &module, &buildlog);

  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    EXPECT_EQ(ZE_RESULT_SUCCESS,
              zeModuleBuildLogGetString(buildlog, &szLog, nullptr));

    char *strLog = (char *)malloc(szLog);
    EXPECT_EQ(ZE_RESULT_SUCCESS,
              zeModuleBuildLogGetString(buildlog, &szLog, strLog));

    std::cerr << "L0 build module failed. Log:\n" << strLog << std::endl;
    free(strLog);
    EXPECT_EQ(ZE_RESULT_SUCCESS, zeModuleBuildLogDestroy(buildlog));
  }

  EXPECT_EQ(ZE_RESULT_SUCCESS, error_no);

  return module;
}

void printModuleKernelName(ze_module_handle_t hModule) {
  uint32_t Count = 0;
  auto ret = zeModuleGetKernelNames(hModule, &Count, nullptr);
  assert(ret == ZE_RESULT_SUCCESS);
  std::unique_ptr<const char *[]> PNames(new const char *[Count]);
  ret = zeModuleGetKernelNames(hModule, &Count, PNames.get());
  assert(ret == ZE_RESULT_SUCCESS);
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    for (uint32_t i = 0; i < Count; ++i) {
      std::cout << std::string(PNames[i]) << std::endl;
    }
  }
}

ze_kernel_handle_t create_function(ze_module_handle_t module,
                                   ze_kernel_flags_t flag,
                                   std::string func_name) {
  ze_kernel_handle_t kernel;
  ze_kernel_desc_t kernel_description = {};
  kernel_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;

  kernel_description.pNext = nullptr;
  kernel_description.flags = flag;
  kernel_description.pKernelName = func_name.c_str();
  auto module_initial = module;
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    std::cout << "create kernel:" << func_name << std::endl;
  }
  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeKernelCreate(module, &kernel_description, &kernel));
  //  EXPECT_EQ(module, module_initial);
  return kernel;
}

ze_kernel_handle_t create_function(ze_module_handle_t module,
                                   std::string func_name) {
  return create_function(module, 0, func_name);
}

std::vector<std::unique_ptr<sycl::kernel>> compiled_kernel;

py::tuple spirv_to_sycl_kernel(sycl::device &device, uint32_t *binary_ptr,
                               size_t binary_size, std::string kernel_name) {

  int32_t n_regs = 0;
  int32_t n_spills = 0;

  auto ctx = device.get_platform().ext_oneapi_get_default_context();
  auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  auto l0_module =
      create_module(l0_context, l0_device, binary_ptr, binary_size);
  printModuleKernelName(l0_module);

  auto l0_kernel = create_function(l0_module, kernel_name);
  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;

  EXPECT_EQ(ZE_RESULT_SUCCESS, zeKernelGetProperties(l0_kernel, &props));

  n_spills = props.spillMemSize;

  auto mod = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                      sycl::bundle_state::executable>(
      {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);

  auto fun = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {mod, l0_kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
      ctx);
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    //  auto kernel_ids = mod.get_kernel_ids();
    //  std::cout << "num_kernels:" << kernel_ids.size() << std::endl;
    //  for (auto& kernel_id : kernel_ids) {
    //    std::cout << "fun name: " << kernel_id.get_name() << std::endl;
    //  }
  }
  compiled_kernel.push_back(std::make_unique<sycl::kernel>(fun));
  sycl::kernel *ptr = compiled_kernel[compiled_kernel.size() - 1].get();
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    std::cout << "compiled kernel ptr: " << ptr << std::endl;
    std::cout << "total kernels:" << compiled_kernel.size() << std::endl;
    for (auto &k : compiled_kernel) {
      std::cout << "  kernel:"
                << k->get_info<sycl::info::kernel::function_name>() << " @"
                << k.get() << std::endl;
    }
  }
  sycl::kernel *k = new sycl::kernel(*ptr);
  py::capsule kernel_capsulle(k, [](void *f) {
    auto kk = static_cast<sycl::kernel *>(f);
    delete kk;
  });
  sycl::kernel_bundle<sycl::bundle_state::executable> *kb =
      new sycl::kernel_bundle<sycl::bundle_state::executable>(mod);
  py::capsule module_capsulle(kb, [](void *f) {
    auto kk =
        static_cast<sycl::kernel_bundle<sycl::bundle_state::executable> *>(f);
    delete kk;
  });
  py::tuple tup =
      py::make_tuple(module_capsulle, kernel_capsulle, n_regs, n_spills);
  return tup;
}

static void register_xpu_device_info(PyObject *module) {
  // Add _DeviceInfo class to intel_extension_for_pytorch._C
  auto m = py::handle(module).cast<py::module>();
}

static inline void *getPointer(const py::object &_obj, int idx) {
  PyObject *obj = _obj.ptr();
  if (PyLong_Check(obj)) {
    auto ptrValue = PyLong_AsUnsignedLongLong(obj);
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    return (void *)ptrValue;
  }
  return (void *)0;
}

PYBIND11_MODULE(sycl_utils, m) {
  m.doc() = "triton sycl utils to load the spirv kernel";
  m.def(
      "get_device_properties",
      [](void *device_ptr) {
        sycl::device *device = static_cast<sycl::device *>(device_ptr);

        auto max_shared_mem =
            device->get_info<sycl::info::device::local_mem_size>();
        bool support_fp64 = device->has(sycl::aspect::fp64);
        auto eu_count_per_ss = device->get_info<
            sycl::info::device::ext_intel_gpu_eu_count_per_subslice>();
        auto threads_per_eu = device->get_info<
            sycl::info::device::ext_intel_gpu_hw_threads_per_eu>();
        auto max_clock_frequency =
            device->get_info<sycl::info::device::max_clock_frequency>();
        auto max_work_group_size =
            device->get_info<sycl::info::device::max_work_group_size>();
        auto max_num_sub_groups =
            device->get_info<sycl::info::device::max_num_sub_groups>();
        auto sub_group_sizes =
            device->get_info<sycl::info::device::sub_group_sizes>();
        auto dev_id =
            device->get_info<sycl::ext::intel::info::device::device_id>();
        auto dev_name = device->get_info<sycl::info::device::name>();

        py::dict properties =
            py::dict("max_shared_mem"_a = max_shared_mem,
                     "support_fp64"_a = support_fp64,
                     "eu_count_per_ss"_a = eu_count_per_ss,
                     "threads_per_eu"_a = threads_per_eu,
                     "max_clock_frequency"_a = max_clock_frequency,
                     "max_work_group_size"_a = max_work_group_size,
                     "max_num_sub_groups"_a = max_num_sub_groups,
                     "sub_group_sizes"_a = sub_group_sizes,
                     "dev_name"_a = dev_name, "dev_id"_a = dev_id);
        return properties;
      },
      "Get the properties for a given device",
      py::return_value_policy::take_ownership);
  m.def(
      "load_binary",
      [](std::string name, py::bytes bytes, int shared, void *device_ptr) {
        std::string binary(bytes);
        sycl::device *device = static_cast<sycl::device *>(device_ptr);
        if (getBoolEnv("MLIR_ENABLE_DUMP")) {
          std::cout << "binary size in u32:" << binary.size() / sizeof(uint32_t)
                    << std::endl;
        }
        return spirv_to_sycl_kernel(*device, (uint32_t *)binary.c_str(),
                                    binary.size() / sizeof(uint32_t), name);
      },
      "Load provided spirv to SYCL kernel",
      py::return_value_policy::take_ownership);
}
