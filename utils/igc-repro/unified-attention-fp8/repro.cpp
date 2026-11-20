// Standalone SPIR-V + Level Zero launcher. No Triton / torch / SYCL dependency.
// Build: g++ -O2 -std=c++17 repro.cpp -I<dir containing level_zero/ze_api.h> -lze_loader -o repro
// Usage: ./repro <case.cfg> [--build-flags "<flags>"] [--iters N] [--warmup N]
#include <level_zero/ze_api.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define ZE(x) do { ze_result_t r_ = (x); if (r_ != ZE_RESULT_SUCCESS) { \
  std::fprintf(stderr, "L0 error 0x%x at %s:%d: %s\n", (unsigned)r_, __FILE__, __LINE__, #x); std::exit(2);} } while (0)

static std::vector<char> readFile(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Tensor { std::string dtype; size_t bytes = 0; std::string init; std::string file; void *dev = nullptr; };
struct Arg { std::string kind, a, b; };

static uint32_t xs = 2463534242u;
static inline uint32_t rnd() { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s case.cfg [--build-flags F] [--iters N] [--warmup N]\n", argv[0]); return 1; }
  std::string cfgPath = argv[1], flags; int iters = 50, warmup = 10;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--build-flags" && i + 1 < argc) flags = argv[++i];
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
  }
  std::string dir = cfgPath.substr(0, cfgPath.find_last_of('/') + 1);
  std::string kname, spv, outName, label; uint32_t grid[3] = {1, 1, 1}, local[3] = {1, 1, 1}; double flops = 0;
  std::map<std::string, Tensor> tensors; std::vector<Arg> args;
  { std::ifstream f(cfgPath); std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line); std::string k; ss >> k;
      if (k == "kernel") ss >> kname; else if (k == "spv") ss >> spv; else if (k == "label") ss >> label;
      else if (k == "grid") ss >> grid[0] >> grid[1] >> grid[2]; else if (k == "local") ss >> local[0] >> local[1] >> local[2];
      else if (k == "flops") ss >> flops; else if (k == "output") ss >> outName;
      else if (k == "tensor") { std::string n; Tensor t; ss >> n >> t.dtype >> t.bytes >> t.init; if (t.init == "file") ss >> t.file; tensors[n] = t; }
      else if (k == "arg") { Arg a; ss >> a.kind >> a.a; tensors.count(a.a); ss >> a.b; args.push_back(a); }
    } }

  ZE(zeInit(0));
  uint32_t nd = 0; ZE(zeDriverGet(&nd, nullptr)); std::vector<ze_driver_handle_t> drivers(nd); ZE(zeDriverGet(&nd, drivers.data()));
  ze_driver_handle_t drv = nullptr; ze_device_handle_t dev = nullptr;
  for (auto d : drivers) { uint32_t n = 0; zeDeviceGet(d, &n, nullptr); std::vector<ze_device_handle_t> ds(n); zeDeviceGet(d, &n, ds.data());
    for (auto x : ds) { ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES}; zeDeviceGetProperties(x, &p); if (p.type == ZE_DEVICE_TYPE_GPU && !dev) { drv = d; dev = x; } } }
  if (!dev) { std::fprintf(stderr, "no Level Zero GPU found\n"); return 3; }
  ze_device_properties_t dp{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES}; ZE(zeDeviceGetProperties(dev, &dp));
  ze_driver_properties_t drp{ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES}; ZE(zeDriverGetProperties(drv, &drp));
  std::printf("device: %s  driverVersion: %u  flags: [%s]\n", dp.name, drp.driverVersion, flags.c_str());

  ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx; ZE(zeContextCreate(drv, &cd, &ctx));
  ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC}; qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_list_handle_t cl; ZE(zeCommandListCreateImmediate(ctx, dev, &qd, &cl));

  // Allocate + initialise tensors.
  for (auto &kv : tensors) {
    Tensor &t = kv.second; ze_device_mem_alloc_desc_t ad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ZE(zeMemAllocDevice(ctx, &ad, t.bytes, 64, dev, &t.dev));
    std::vector<unsigned char> h(t.bytes, 0);
    if (t.init == "file") { auto d = readFile(dir + t.file); std::memcpy(h.data(), d.data(), std::min(d.size(), h.size())); }
    else if (t.init == "rand_fp8") for (auto &b : h) { unsigned char v = rnd() & 0xFF; if ((v & 0x7F) == 0x7F) v &= 0xFE; b = v; }  // avoid e4m3 NaN
    else if (t.init == "rand_bf16") for (size_t i = 0; i + 1 < h.size(); i += 2) { uint32_t r = rnd(); uint16_t v = (uint16_t)(0x3F00 | (r & 0x7F)) | (uint16_t)((r >> 8) & 1) << 15; std::memcpy(&h[i], &v, 2); }
    ZE(zeCommandListAppendMemoryCopy(cl, t.dev, h.data(), t.bytes, nullptr, 0, nullptr)); ZE(zeCommandListHostSynchronize(cl, UINT64_MAX));
  }

  auto il = readFile(dir + spv);
  ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC}; md.format = ZE_MODULE_FORMAT_IL_SPIRV; md.inputSize = il.size();
  md.pInputModule = (const uint8_t *)il.data(); md.pBuildFlags = flags.c_str();
  ze_module_handle_t mod; ze_module_build_log_handle_t log = nullptr; ze_result_t mr = zeModuleCreate(ctx, dev, &md, &mod, &log);
  if (mr != ZE_RESULT_SUCCESS) { size_t n = 0; if (log) { zeModuleBuildLogGetString(log, &n, nullptr); std::string s(n, 0); zeModuleBuildLogGetString(log, &n, s.data()); std::fprintf(stderr, "build log: %s\n", s.c_str()); }
    std::fprintf(stderr, "zeModuleCreate failed 0x%x\n", (unsigned)mr); return 4; }
  ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC}; kd.pKernelName = kname.c_str(); ze_kernel_handle_t k; ZE(zeKernelCreate(mod, &kd, &k));
  ZE(zeKernelSetGroupSize(k, local[0], local[1], local[2]));
  std::vector<int32_t> i32s(args.size()); std::vector<int64_t> i64s(args.size()); std::vector<float> f32s(args.size()); std::vector<void *> ptrs(args.size(), nullptr);
  for (size_t i = 0; i < args.size(); i++) { const Arg &a = args[i];
    if (a.kind == "tensor") { ptrs[i] = tensors.at(a.a).dev; ZE(zeKernelSetArgumentValue(k, i, sizeof(void *), &ptrs[i])); }
    else if (a.kind == "null") ZE(zeKernelSetArgumentValue(k, i, sizeof(void *), &ptrs[i]));
    else if (a.kind == "i32") { i32s[i] = (int32_t)std::strtol(a.a.c_str(), nullptr, 10); ZE(zeKernelSetArgumentValue(k, i, 4, &i32s[i])); }
    else if (a.kind == "i64") { i64s[i] = std::strtoll(a.a.c_str(), nullptr, 10); ZE(zeKernelSetArgumentValue(k, i, 8, &i64s[i])); }
    else if (a.kind == "f32") { f32s[i] = std::strtof(a.a.c_str(), nullptr); ZE(zeKernelSetArgumentValue(k, i, 4, &f32s[i])); }
    else { std::fprintf(stderr, "bad arg kind %s\n", a.kind.c_str()); return 1; } }

  ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC}; epd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP; epd.count = 1;
  ze_event_pool_handle_t ep; ZE(zeEventPoolCreate(ctx, &epd, 0, nullptr, &ep)); ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC}; ze_event_handle_t ev; ZE(zeEventCreate(ep, &ed, &ev));
  ze_group_count_t gc{grid[0], grid[1], grid[2]};
  std::vector<double> ms;
  for (int i = 0; i < warmup + iters; i++) {
    ZE(zeEventHostReset(ev)); ZE(zeCommandListAppendLaunchKernel(cl, k, &gc, ev, 0, nullptr)); ZE(zeEventHostSynchronize(ev, UINT64_MAX));
    ze_kernel_timestamp_result_t ts; ZE(zeEventQueryKernelTimestamp(ev, &ts));
    uint64_t mask = dp.kernelTimestampValidBits >= 64 ? ~0ull : ((1ull << dp.kernelTimestampValidBits) - 1);
    uint64_t ticks = (ts.context.kernelEnd - ts.context.kernelStart) & mask;
    double res = (double)dp.timerResolution; double ns = res > 1e6 ? ticks * (1e9 / res) : ticks * res;   // ns/tick (old) or Hz (new spec)
    if (i >= warmup) ms.push_back(ns * 1e-6);
  }
  std::sort(ms.begin(), ms.end());
  double med = ms[ms.size() / 2], mn = ms.front();
  // Output checksum: sum of the raw bytes interpreted as a polynomial hash (bit-exact comparison across drivers).
  uint64_t h = 1469598103934665603ull;
  if (!outName.empty()) { Tensor &t = tensors.at(outName); std::vector<unsigned char> o(t.bytes);
    ZE(zeCommandListAppendMemoryCopy(cl, o.data(), t.dev, t.bytes, nullptr, 0, nullptr)); ZE(zeCommandListHostSynchronize(cl, UINT64_MAX));
    for (unsigned char b : o) { h ^= b; h *= 1099511628211ull; } }
  std::printf("RESULT label=%s flags=[%s] p50_ms=%.4f min_ms=%.4f tflops_p50=%.3f tflops_min=%.3f checksum=%016llx\n", label.c_str(), flags.c_str(),
              med, mn, flops > 0 ? flops * 1e-12 / (med * 1e-3) : 0.0, flops > 0 ? flops * 1e-12 / (mn * 1e-3) : 0.0, (unsigned long long)h);
  return 0;
}
