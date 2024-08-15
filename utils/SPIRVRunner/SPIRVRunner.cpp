#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <torch/torch.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "sycl_functions.h"

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

auto load_tensor(const std::string &filename) {
  std::ifstream ins(filename, std::ios::binary);
  if (!ins.is_open()) {
    throw std::runtime_error("Failed to open file " + filename);
  }

  ins.seekg(0, std::ios::end);
  auto fileSize = ins.tellg();

  std::vector<char> bytes(fileSize);
  ins.seekg(0, std::ios::beg);
  ins.read(bytes.data(), fileSize);

  return torch::pickle_load(bytes).toTensor();
}

void write_tensor(const std::string &filename, torch::Tensor &tensor) {
  std::ofstream outs(filename, std::ios::binary | std::ios::trunc);
  auto output_bytes = torch::pickle_save(tensor);

  outs.write(output_bytes.data(), output_bytes.size());
}

std::vector<char> read_spirv(const std::string &filename) {
  std::ifstream ins(filename, std::ios::binary);
  if (!ins.is_open()) {
    throw std::runtime_error("Failed to open file " + filename);
  }

  ins.seekg(0, std::ios::end);
  auto fileSize = ins.tellg();

  std::vector<char> bytes(fileSize);
  ins.seekg(0, std::ios::beg);
  ins.read(bytes.data(), fileSize);

  return bytes;
}

/** SYCL Globals **/

SyclQueueMap g_sycl_queue_map;

static std::vector<ze_device_handle_t> g_devices;
static std::vector<std::pair<sycl::device, ze_device_handle_t>>
    g_sycl_l0_device_list;

static inline void gpuAssert(ze_result_t code) {
  if (code != ZE_RESULT_SUCCESS) {
    auto str = parseZeResultCode(code);
    throw std::runtime_error(str);
  }
}

template <typename T>
static inline T checkSyclErrors(const std::tuple<T, ze_result_t> tuple) {
  gpuAssert(std::get<1>(tuple));
  return std::get<0>(tuple);
}

/** SYCL Functions **/
std::tuple<std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>>,
           std::unique_ptr<sycl::kernel>, int32_t, int32_t>
loadBinary(const std::string &kernel_name, uint8_t *binary_ptr,
           const size_t binary_size, const size_t deviceId) {
  int32_t n_regs = 0;
  int32_t n_spills = 0;

  if (!(deviceId < g_sycl_l0_device_list.size())) {
    throw std::runtime_error("Device is not found " + std::to_string(deviceId));
  }

  const auto &sycl_l0_device_pair = g_sycl_l0_device_list[deviceId];
  const sycl::device sycl_device = sycl_l0_device_pair.first; // TODO: reference ok? 

  const auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();
  const auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
  const auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  const char *build_flags = "";
  auto l0_module = checkSyclErrors(create_module(
      l0_context, l0_device, binary_ptr, binary_size, build_flags));
  auto l0_kernel = checkSyclErrors(create_function(l0_module, kernel_name));

  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  gpuAssert((zeKernelGetProperties(l0_kernel, &props)));
  n_spills = props.spillMemSize;
  auto mod = std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                      sycl::bundle_state::executable>(
      {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx));
  auto fun = std::make_unique<sycl::kernel>(sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {*mod, l0_kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
      ctx));

  return std::make_tuple(std::move(mod), std::move(fun), n_regs, n_spills);
}

ze_context_handle_t initContext(sycl::queue *sycl_queue) {
  if (g_sycl_queue_map.find(*sycl_queue) == g_sycl_queue_map.end()) {
    const auto updated_sycl_devices = update(*sycl_queue, g_sycl_queue_map);
    if (!updated_sycl_devices.empty()) {
      // Update global data
      const uint32_t deviceCount =
          std::min(updated_sycl_devices.size(), g_devices.size());
      for (uint32_t i = 0; i < deviceCount; ++i) {
        g_devices[i] = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            updated_sycl_devices[i]);
      }
    }
  }
  auto context = g_sycl_queue_map[*sycl_queue].context;
  return context;
}

size_t initDevices(sycl::queue *sycl_queue) {
  auto sycl_context = sycl_queue->get_context();

  // Get sycl-device
  const std::vector<sycl::device> &sycl_devices = sycl_context.get_devices();

  // Retrieve l0 devices
  const uint32_t deviceCount = sycl_devices.size();
  for (uint32_t i = 0; i < deviceCount; ++i) {
    g_sycl_l0_device_list.push_back(std::make_pair(
        sycl_devices[i], sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                             sycl_devices[i])));
    g_devices.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        sycl_devices[i]));
  }

  return deviceCount;
}

static void set_scalar_arg(sycl::handler &cgh, int index, size_t size,
                           const void *value) {
  switch (size) {
  case sizeof(uint8_t):
    cgh.set_arg(index, *static_cast<const uint8_t *>(value));
    break;
  case sizeof(uint16_t):
    cgh.set_arg(index, *static_cast<const uint16_t *>(value));
    break;
  case sizeof(uint32_t):
    cgh.set_arg(index, *static_cast<const uint32_t *>(value));
    break;
  case sizeof(uint64_t):
    cgh.set_arg(index, *static_cast<const uint64_t *>(value));
    break;
  default:
    assert(false && "wrong scalar size in sycl gen.");
  }
}

static void sycl_kernel_launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                               int num_warps, int threads_per_warp,
                               int shared_memory, sycl::queue &stream,
                               sycl::kernel &kernel_ptr, void *arg0, void *arg1,
                               void *arg2, int32_t arg3) {
  std::string kernel_name =
      kernel_ptr.get_info<sycl::info::kernel::function_name>();
  void *params[] = {&arg0, &arg1, &arg2, &arg3};
  uint32_t num_params = sizeof(params) / sizeof(params[0]);
  uint32_t expected_num_params =
      kernel_ptr.get_info<sycl::info::kernel::num_args>();
  size_t global_range_x = gridX * threads_per_warp * num_warps;
  size_t global_range_y = gridY;
  size_t global_range_z = gridZ;
  size_t local_range_x = num_warps * threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;
  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);
  if (shared_memory) {
    expected_num_params -= 1;
  }
  assert(num_params == expected_num_params &&
         "number of kernel param not matched");

  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {
    set_scalar_arg(cgh, 0, sizeof(void *), params[0]);
    set_scalar_arg(cgh, 1, sizeof(void *), params[1]);
    set_scalar_arg(cgh, 2, sizeof(void *), params[2]);
    set_scalar_arg(cgh, 3, sizeof(int32_t), params[3]);
    if (shared_memory) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
      cgh.set_arg(num_params, local_buffer);
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    } else {
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    }
  };
  auto event = stream.submit(cgf);

  stream.wait();
}

at::Tensor launchKernel(sycl::queue stream, sycl::kernel kernel,
                        const torch::Tensor &a, const torch::Tensor &b) {
  int gridX = 97;
  int gridY = 1;
  int gridZ = 1;

  int num_warps = 4;
  int num_ctas = 1;
  int shared_memory = 0;
  int threads_per_warp = 32;

  int _arg3 = 98432;
  
  torch::Tensor output =
      torch::zeros({a.sizes()[0]}, c10::nullopt,
                   at::TensorOptions{c10::ScalarType::Float}); 
   std::cout << "Tensor output: " << output.sizes() << ", " << output.scalar_type() << " (" << output.nbytes() << " bytes)"
            << std::endl;

  auto tensor_ptr = [](const torch::Tensor &t) -> void * {
    return reinterpret_cast<void *>(t.data_ptr());
  };

  auto a_dev = sycl::malloc_device<char>(a.nbytes(), stream);
  auto b_dev = sycl::malloc_device<char>(b.nbytes(), stream);
  auto output_dev = sycl::malloc_device<char>(output.nbytes(), stream);

  stream.submit([&](sycl::handler &cgh) {
    cgh.memcpy(a_dev, tensor_ptr(a), a.nbytes());
  });
  stream.submit([&](sycl::handler &cgh) {
    cgh.memcpy(b_dev, tensor_ptr(b), b.nbytes());
  });
  stream.submit([&](sycl::handler &cgh) {
    cgh.memcpy(output_dev, tensor_ptr(output), output.nbytes());
  });
  stream.wait();

  sycl_kernel_launch(gridX, gridY, gridZ, num_warps, threads_per_warp,
                     shared_memory, stream, kernel, a_dev,
                     b_dev, output_dev, _arg3);


  // copy back
  stream.submit([&](sycl::handler &cgh) {
    cgh.memcpy(tensor_ptr(output), output_dev, output.nbytes());
  });
  stream.wait();

  return output;
}

int main() {
  // initialize sycl runtime
  sycl::default_selector d_selector;
  sycl::queue q = sycl::queue(d_selector, exception_handler);
  
  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  auto context = initContext(&q);
  auto device_count = initDevices(&q);

  auto a = load_tensor("x.pt");
  auto b = load_tensor("y.pt");
  std::cout << "Tensor a: " << a.sizes() << ", " << a.scalar_type() << " (" << a.nbytes() << " bytes)"
            << std::endl;
  std::cout << "Tensor b: " << b.sizes() << ", " << b.scalar_type() << " (" << b.nbytes() << " bytes)"
            << std::endl;

  // read spirv
  auto spirv = read_spirv("add_kernel.spv");
  std::cout << "Read " << spirv.size() << " byte kernel." << std::endl;

  auto [kernel_bundle, kernel, n_regs, n_spills] =
      loadBinary("add_kernel", reinterpret_cast<uint8_t *>(spirv.data()),
                 spirv.size() / sizeof(uint32_t), 0);

  // TODO: missing number of registers 
  std::cout << "Loaded kernel with " << n_regs << " registers and " << n_spills
            << " register spills." << std::endl;

  auto output = launchKernel(q, *kernel, a, b);
  std::cout << "Kernel return output: " << output[0] << std::endl;
  write_tensor("cpp_outs.pt", output);
}
