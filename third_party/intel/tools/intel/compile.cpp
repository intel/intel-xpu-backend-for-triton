/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <sstream>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

// helpers to check for ze errors
#define ZE_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert(ze_result_t code, const char *file, int line) {{
  if (code != ZE_RESULT_SUCCESS) {{
    const char *prefix = "Triton Error [ZE]: ";
    std::string str = std::to_string(code);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str.c_str());
    printf("%s\\n", err);
    exit(code);
  }}
}}

// ze globals
#define SPV_NAME {kernel_name}_spv
ze_module_handle_t {kernel_name}_mod = NULL;
ze_kernel_handle_t {kernel_name}_func = NULL;
unsigned char SPV_NAME[{bin_size}] = {{ {bin_data} }};
// sycl globals
const sycl::device sycl_device;
const auto ctx =
    sycl_device.get_platform().ext_oneapi_get_default_context();
const auto l0_device =
    sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
const auto l0_context =
    sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

void unload_{kernel_name}(void) {{
    // Not implemeted
}}

void load_{kernel_name}() {{
    uint8_t *binary_ptr = (uint8_t *)&SPV_NAME;
    size_t binary_size = {bin_size};

    const bool is_spv = {is_spv};

    ze_module_desc_t module_description {{}};
    module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    module_description.format = is_spv ? ZE_MODULE_FORMAT_IL_SPIRV : ZE_MODULE_FORMAT_NATIVE;
    module_description.inputSize = static_cast<uint32_t>(binary_size);
    module_description.pInputModule = binary_ptr;
    module_description.pBuildFlags = "{build_flags}";
    ZE_CHECK(zeModuleCreate(l0_context, l0_device, &module_description, &{kernel_name}_mod, nullptr));

    ze_kernel_desc_t kernel_description {{}};
    kernel_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernel_description.pNext = nullptr;
    kernel_description.flags = ZE_KERNEL_FLAG_FORCE_RESIDENCY;
    kernel_description.pKernelName = "{triton_kernel_name}";
    ZE_CHECK(zeKernelCreate({kernel_name}_mod, &kernel_description, &{kernel_name}_func));

    ze_kernel_properties_t props;
    props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    props.pNext = nullptr;
    ZE_CHECK(zeKernelGetProperties({kernel_name}_func, &props));
}}

/*
{kernel_docstring}
*/
template <class T>
static inline void set_scalar_arg(sycl::handler &cgh, int index, const void *value) {{
  cgh.set_arg(index, *static_cast<const T *>(value));
}}

static inline void set_argument(sycl::handler &cgh, int index, const std::string type, const void *value) {{
  if (type == "int32_t") {{
    set_scalar_arg<int32_t>(cgh, index, value);
  }} else if (type == "int64_t") {{
    set_scalar_arg<int64_t>(cgh, index, value);
  }} else if (type == "uint32_t") {{
    set_scalar_arg<uint32_t>(cgh, index, value);
  }} else if (type == "uint8_t") {{
    set_scalar_arg<uint8_t>(cgh, index, value);
  }} else if (type == "uint16_t") {{
    set_scalar_arg<uint16_t>(cgh, index, value);
  }} else if (type == "uint64_t") {{
    set_scalar_arg<uint64_t>(cgh, index, value);
  }} else if (type == "float") {{
    set_scalar_arg<float>(cgh, index, value);
  }} else if (type == "double") {{
    set_scalar_arg<double>(cgh, index, value);
  }} else if (type == "int16_t") {{
    set_scalar_arg<int16_t>(cgh, index, value);
  }} else if (type == "int8_t") {{
    set_scalar_arg<int8_t>(cgh, index, value);
  }} else if (type == "void*") {{
    set_scalar_arg<void*>(cgh, index, value);
  }} else {{
    throw std::runtime_error("Argument type doesnt match");
  }}
}}

int32_t {kernel_name}(sycl::queue &stream, {signature}) {{
  auto sycl_mod = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                      sycl::bundle_state::executable>(
      {{{kernel_name}_mod, sycl::ext::oneapi::level_zero::ownership::transfer}}, ctx);
  auto sycl_kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {{sycl_mod, {kernel_name}_func, sycl::ext::oneapi::level_zero::ownership::transfer}},
      ctx);
  std::string kernel_name = sycl_kernel.get_info<sycl::info::kernel::function_name>();
  std::string driver_version = stream.get_device().get_info<sycl::info::device::driver_version>();
  void *params[] = {{ {arg_pointers} }};
  uint32_t num_params = sizeof(params)/sizeof(params[0]);
  uint32_t expected_num_params = sycl_kernel.get_info<sycl::info::kernel::num_args>();

  size_t global_range_x = {gridX} * {threads_per_warp} * {num_warps};
  size_t global_range_y = {gridY};
  size_t global_range_z = {gridZ};
  size_t local_range_x = {num_warps} * {threads_per_warp};
  if (driver_version.find("+") != std::string::npos) {{
    local_range_x = 16;
  }}
  size_t local_range_y = 1;
  size_t local_range_z = 1;

  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);

  if (static_cast<bool>({shared})) {{
    expected_num_params -= 1;
  }}
  assert(num_params == expected_num_params && "number of kernel param not matched");
  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {{
    int idx = 0;
    std::string types = std::string("{arg_types}");
    std::istringstream sstream(types);
    std::string type;
    while (std::getline(sstream, type, ',')) {{
        size_t first = type.find_first_not_of(" \t");
        size_t last = type.find_last_not_of(" \t");
        if (first != std::string::npos && last != std::string::npos) {{
            type = type.substr(first, last - first + 1);
        }}
        set_argument(cgh, idx, type, params[idx]);
        idx++;
    }}
    if (static_cast<bool>({shared})) {{
        using share_mem_t = sycl::local_accessor<int8_t, 1>;
        share_mem_t local_buffer = share_mem_t({shared}, cgh);
        cgh.set_arg(num_params, local_buffer);
        cgh.parallel_for(parallel_work_size, sycl_kernel);
    }} else {{
        cgh.parallel_for(parallel_work_size, sycl_kernel);
    }}
  }};
  stream.submit(cgf);
  stream.wait_and_throw();
  return 0;
}}
