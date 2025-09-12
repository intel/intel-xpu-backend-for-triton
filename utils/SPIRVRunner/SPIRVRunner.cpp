#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <torch/torch.h>

#include "llvm_parser.h"
#include "sycl_functions.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

auto read_file_as_bytes(const std::string &filename) {
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

auto load_tensor(const std::string &filename) {
  auto bytes = read_file_as_bytes(filename);
  return torch::pickle_load(bytes).toTensor();
}

void write_tensor(const std::string &filename, torch::Tensor &tensor) {
  std::ofstream outs(filename, std::ios::binary | std::ios::trunc);
  auto output_bytes = torch::pickle_save(tensor);
  outs.write(output_bytes.data(), output_bytes.size());
}

auto read_spirv(const std::string &filename) {
  return read_file_as_bytes(filename);
}

// Host output tensor buffers and indexes
struct TensorBuffer {
  torch::Tensor buffer_ptr;
  size_t index;
};

// Structure that contains Triton kernel arguments
struct KernelArguments {
  int gridX;
  int gridY;
  int gridZ;
  int num_ctas;
  int num_stages;
  int num_warps;
  int threads_per_warp;
  int shared_memory;
  std::string kernel_name;
  std::string build_flags;
  std::string spv_name;
  ordered_json jsonData;
  std::vector<char *> dev_buffers;
  std::vector<TensorBuffer> host_outbuffers;
  std::vector<std::string> out_tensor_names;
  std::string spirv_dump_dir;

  KernelArguments(const std::vector<std::string> &outtensornames) {
    // Check if the triton_xpu_dump path exists if not point to current
    // directory
    auto env_path = std::getenv("TRITON_XPU_DUMP_SPIRV_KERNEL_ARGS");
    spirv_dump_dir = (env_path != nullptr)
                         ? env_path
                         : std::filesystem::current_path().string();
    if (std::filesystem::exists(spirv_dump_dir)) {
      std::ifstream file(spirv_dump_dir + "/args_data.json");
      if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file");
      }
      file >> jsonData;
      if (jsonData.is_discarded()) {
        throw std::runtime_error("Invalid JSON format in the file");
      }
      file.close();
    } else {
      throw std::runtime_error("Triton Spirv dump path doesnt exist");
    }

    gridX = jsonData.at("gridX");
    gridY = jsonData.at("gridY");
    gridZ = jsonData.at("gridZ");
    num_warps = jsonData.at("num_warps");
    shared_memory = jsonData.at("shared_memory");
    threads_per_warp = jsonData.at("threads_per_warp");
    kernel_name = jsonData.at("kernel_name");
    build_flags = jsonData.at("build_flags");
    spv_name =
        spirv_dump_dir + "/" + jsonData.at("spv_name").get<std::string>();
    out_tensor_names = outtensornames;
  }
};

/** SYCL Globals **/
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

sycl::context get_default_context(const sycl::device &sycl_device) {
  const auto &platform = sycl_device.get_platform();
#ifdef WIN32
  sycl::context ctx;
  try {
#if __SYCL_COMPILER_VERSION >= 20250604
    ctx = platform.khr_get_default_context();
#else
    ctx = platform.ext_oneapi_get_default_context();
#endif
  } catch (const std::runtime_error &ex) {
    // This exception is thrown on Windows because
    // khr_get_default_context is not implemented. But it can be safely
    // ignored it seems.
#if _DEBUG
    std::cout << "ERROR: " << ex.what() << std::endl;
#endif
  }
  return ctx;
#else
#if __SYCL_COMPILER_VERSION >= 20250604
  return platform.khr_get_default_context();
#else
  return platform.ext_oneapi_get_default_context();
#endif
#endif
}

/** SYCL Functions **/
std::tuple<sycl::kernel_bundle<sycl::bundle_state::executable>, sycl::kernel,
           int32_t, int32_t>
loadBinary(const std::string &kernel_name, const std::string &build_flags,
           uint8_t *binary_ptr, const size_t binary_size,
           const size_t deviceId) {
  int32_t n_regs = 0;
  int32_t n_spills = 0;

  if (!(deviceId < g_sycl_l0_device_list.size())) {
    throw std::runtime_error("Device is not found " + std::to_string(deviceId));
  }

  const auto &sycl_l0_device_pair = g_sycl_l0_device_list[deviceId];
  const sycl::device sycl_device = sycl_l0_device_pair.first;

  const auto &ctx = get_default_context(sycl_device);

  const auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
  const auto l0_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  ze_module_build_log_handle_t buildlog;
  auto l0_module = checkSyclErrors(
      create_module(l0_context, l0_device, binary_ptr, binary_size, &buildlog,
                    build_flags.c_str()));
  auto l0_kernel =
      checkSyclErrors(create_function(l0_module, kernel_name, &buildlog));

  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  gpuAssert((zeKernelGetProperties(l0_kernel, &props)));
  n_spills = props.spillMemSize;
  auto mod = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                      sycl::bundle_state::executable>(
      {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);
  auto fun = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {mod, l0_kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
      ctx);

  return std::make_tuple(mod, fun, n_regs, n_spills);
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
  }

  return deviceCount;
}

template <class T>
static inline void set_scalar_arg(sycl::handler &cgh, int index,
                                  const void *value) {
  cgh.set_arg(index, *static_cast<const T *>(value));
}

void set_argument(sycl::handler &cgh, int index, ordered_json &item) {
  auto type = item.at("ctype").get<std::string>();
  if (type == "i32" || type == "i1") {
    auto val = item.at("value").get<int32_t>();
    set_scalar_arg<int32_t>(cgh, index, &val);
  } else if (type == "i64") {
    auto val = item.at("value").get<int64_t>();
    set_scalar_arg<int64_t>(cgh, index, &val);
  } else if (type == "u1" || type == "u32") {
    auto val = item.at("value").get<uint32_t>();
    set_scalar_arg<uint32_t>(cgh, index, &val);
  } else if (type == "u8") {
    auto val = item.at("value").get<uint8_t>();
    set_scalar_arg<uint8_t>(cgh, index, &val);
  } else if (type == "u16") {
    auto val = item.at("value").get<uint16_t>();
    set_scalar_arg<uint16_t>(cgh, index, &val);
  } else if (type == "u64") {
    auto val = item.at("value").get<uint64_t>();
    set_scalar_arg<uint64_t>(cgh, index, &val);
  } else if (type == "fp32" || type == "f32") {
    auto val = item.at("value").get<float>();
    set_scalar_arg<float>(cgh, index, &val);
  } else if (type == "fp64") {
    auto val = item.at("value").get<double>();
    set_scalar_arg<double>(cgh, index, &val);
  } else if (type == "i16") {
    auto val = item.at("value").get<int16_t>();
    set_scalar_arg<int16_t>(cgh, index, &val);
  } else if (type == "i8") {
    auto val = item.at("value").get<int8_t>();
    set_scalar_arg<int8_t>(cgh, index, &val);
  } else {
    throw std::runtime_error("Argument type doesnt match");
  }
}

static void sycl_kernel_launch(sycl::queue &stream, sycl::kernel &kernel_ptr,
                               KernelArguments triton_args,
                               bool get_kernel_time) {
  std::string kernel_name =
      kernel_ptr.get_info<sycl::info::kernel::function_name>();

  [[maybe_unused]] uint32_t expected_num_params =
      kernel_ptr.get_info<sycl::info::kernel::num_args>();

  size_t global_range_x =
      triton_args.gridX * triton_args.threads_per_warp * triton_args.num_warps;
  size_t global_range_y = triton_args.gridY;
  size_t global_range_z = triton_args.gridZ;
  size_t local_range_x = triton_args.num_warps * triton_args.threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;

  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);

  if (triton_args.shared_memory) {
    expected_num_params -= 1;
  }
  int tensorIdx = 0;
  uint32_t narg = 0;
  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {
    // Loop below loads the kernel arguments tensors/sclars
    for (auto &item : triton_args.jsonData["argument_list"]) {
      if (item.contains("type")) {
        if (item.at("type").get<std::string>() == "tensor") {
          set_scalar_arg<void *>(
              cgh, narg++,
              static_cast<void *>(&triton_args.dev_buffers.at(tensorIdx++)));
        } else
          set_argument(cgh, narg++, item);
      } else {
        throw std::runtime_error(
            "Type entry is missing in JSON argument_list\n");
      }
    }
    if (narg != expected_num_params) {
      // global scratch.
      cgh.set_arg(narg++, nullptr);
      // profile scratch
      cgh.set_arg(narg++, nullptr);
    }
    if (triton_args.shared_memory) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t local_buffer = share_mem_t(triton_args.shared_memory, cgh);
      cgh.set_arg(narg, local_buffer);
    }
    assert(narg == expected_num_params);
    cgh.parallel_for(parallel_work_size, kernel_ptr);
  };
  if (get_kernel_time) {
    sycl::event event = stream.submit(cgf);
    event.wait();
    uint64_t start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / 1000000;
    std::cout << "Kernel execution time: " << duration << " ms" << std::endl;
  } else {
    stream.submit(cgf);
  }
  stream.wait_and_throw();
}

at::TensorOptions getTensorOptions(const std::string &dtype) {
  if (dtype == "torch.float32") {
    return at::TensorOptions{c10::ScalarType::Float};
  } else if (dtype == "torch.float64") {
    return at::TensorOptions{c10::ScalarType::Double};
  } else if (dtype == "torch.float16" || dtype == "torch.half") {
    return at::TensorOptions{c10::ScalarType::Half};
  } else if (dtype == "torch.uint8") {
    return at::TensorOptions{c10::ScalarType::Byte};
  } else if (dtype == "torch.int8") {
    return at::TensorOptions{c10::ScalarType::Char};
  } else if (dtype == "torch.int16" || dtype == "torch.short") {
    return at::TensorOptions{c10::ScalarType::Short};
  } else if (dtype == "torch.int32" || dtype == "torch.int") {
    return at::TensorOptions{c10::ScalarType::Int};
  } else if (dtype == "torch.int64" || dtype == "torch.long") {
    return at::TensorOptions{c10::ScalarType::Long};
  } else {
    return at::TensorOptions();
  }
}

void validate_results(std::vector<TensorBuffer> &output_tensors,
                      const std::vector<std::string> &expected_outputs,
                      const std::string &spirv_dump_dir) {
  if (output_tensors.size() != expected_outputs.size()) {
    throw std::runtime_error(
        "Output tensors and expected outputs size mismatch");
  }
  auto idx = 0;
  std::cout << "Validating results..." << std::endl;
  for (const auto &expected_output : expected_outputs) {
    std::string expected_output_path = spirv_dump_dir + "/" + expected_output;
    torch::Tensor expected_tensor = load_tensor(expected_output_path);
    torch::Tensor actual_tensor = output_tensors[idx++].buffer_ptr;

    if (expected_tensor.sizes() != actual_tensor.sizes()) {
      throw std::runtime_error("Size mismatch");
    }

    if (expected_tensor.dtype() != actual_tensor.dtype()) {
      throw std::runtime_error("Dtype mismatch");
    }

    if (!torch::allclose(expected_tensor, actual_tensor)) {
      throw std::runtime_error("Tensors are not close enough");
    }
  }
  std::cout << "Validation successful!" << std::endl;
}

std::vector<TensorBuffer> launchKernel(sycl::queue stream, sycl::kernel kernel,
                                       KernelArguments triton_args,
                                       bool get_kernel_time) {

  auto tensor_ptr = [](const torch::Tensor &t) -> void * {
    return static_cast<void *>(t.data_ptr());
  };

  for (auto &item : triton_args.jsonData["argument_list"]) {
    if (item.contains("type")) {
      if (item.at("type").get<std::string>() == "tensor") {
        auto tensor_name = triton_args.spirv_dump_dir + "/" +
                           item.at("name").get<std::string>() + ".pt";
        auto tensor = load_tensor(tensor_name);
        auto nbytes = tensor.nbytes();
        char *dev = nullptr;
        if (nbytes) {
          dev = sycl::malloc_device<char>(nbytes, stream);
          if (!dev)
            throw std::runtime_error("Device Memory Allocation Failed \n");
        }
        triton_args.dev_buffers.push_back(dev);
        if (nbytes)
          stream.memcpy(dev, tensor_ptr(tensor), nbytes).wait_and_throw();

        // Configure output tensor
        if (std::find(triton_args.out_tensor_names.begin(),
                      triton_args.out_tensor_names.end(),
                      item.at("name").get<std::string>()) !=
            triton_args.out_tensor_names.end()) {
          TensorBuffer tb;
          tb.buffer_ptr = torch::zeros({tensor.sizes()},
                                       getTensorOptions(item.at("dtype")));
          tb.index = triton_args.dev_buffers.size() - 1;
          triton_args.host_outbuffers.push_back(tb);
          std::cout
              << "Tensor output[" << triton_args.host_outbuffers.back().index
              << "]: " << triton_args.host_outbuffers.back().buffer_ptr.sizes()
              << ", "
              << triton_args.host_outbuffers.back().buffer_ptr.scalar_type()
              << " (" << triton_args.host_outbuffers.back().buffer_ptr.nbytes()
              << " bytes)" << std::endl;
        }
      }
    } else {
      throw std::runtime_error("Type entry is missing in JSON argument_list");
    }
  }

  // Launch SYCL kernel
  sycl_kernel_launch(stream, kernel, triton_args, get_kernel_time);

  // copy back the output tensors
  for (const auto &item : triton_args.host_outbuffers) {
    stream
        .memcpy(tensor_ptr(item.buffer_ptr),
                triton_args.dev_buffers.at(item.index),
                item.buffer_ptr.nbytes())
        .wait_and_throw();
  }

  for (auto *dev_ptr : triton_args.dev_buffers) {
    if (dev_ptr)
      sycl::free(dev_ptr, stream);
  }

  return triton_args.host_outbuffers;
}

int main(int argc, char **argv) {
  try {
    command_line_parser cli(argc, argv);
    auto cliopts = cli.parse();

    // initialize sycl runtime
    sycl::queue q;
    if (cliopts.get_kernel_time) {
      sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
      q = sycl::queue(sycl::gpu_selector_v, exception_handler, prop_list);
    } else {
      q = sycl::queue(sycl::gpu_selector_v, exception_handler);
    }

    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    initDevices(&q);

    // Parse the JSON file and create argument dictionary
    KernelArguments tritonArgDict(cliopts.output_tensors);

    // read spirv
    auto spirv = read_spirv(tritonArgDict.spv_name);
    std::cout << "Read " << spirv.size() << " byte kernel." << std::endl;

    auto [kernel_bundle, kernel, n_regs, n_spills] =
        loadBinary(tritonArgDict.kernel_name, tritonArgDict.build_flags,
                   reinterpret_cast<uint8_t *>(spirv.data()), spirv.size(), 0);

    // TODO: missing number of registers
    std::cout << "Loaded kernel with " << n_regs << " registers and "
              << n_spills << " register spills." << std::endl;

    auto output_tensors =
        launchKernel(q, kernel, tritonArgDict, cliopts.get_kernel_time);

    // Write output tensors to file
    for (auto &item : output_tensors) {
      auto output_tensor = tritonArgDict.spirv_dump_dir + "/cpp_outs_" +
                           std::to_string(item.index) + ".pt";
      write_tensor(output_tensor, item.buffer_ptr);
      std::cout << "Output Tensor Path: " << output_tensor << std::endl;
    }

    // Validate results
    if (!cliopts.validate_results.empty()) {
      validate_results(output_tensors, cliopts.validate_results,
                       tritonArgDict.spirv_dump_dir);
    }
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
