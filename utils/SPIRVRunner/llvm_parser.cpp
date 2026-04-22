#include "llvm_parser.h"

command_line_parser::command_line_parser(int argc, char **argv)
    : argc(argc), argv(argv) {}

command_line_parser::options command_line_parser::parse() {
  options opts;
  llvm::cl::list<std::string> output_tensors(
      "o",
      llvm::cl::desc(
          "<Specify Output Tensor Names (Ex: -o tensor_1,tensor_2 or skip)>"),
      llvm::cl::CommaSeparated);
  llvm::cl::opt<bool> enable_profiling(
      "p", llvm::cl::desc("Enable kernel time profiling"),
      llvm::cl::init(opts.get_kernel_time));

  llvm::cl::list<std::string> validate_results(
      "v",
      llvm::cl::desc("<Specify Expected Output Tensor Names (Ex: -v "
                     "expected_tensor1.pt,expected_tensor2.pt or skip)>"),
      llvm::cl::CommaSeparated);
  llvm::cl::opt<std::string> spirv_dump_dir(
      "d", llvm::cl::desc("<Specify SPIRV dump directory path>"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "SPIRVRunner\n");

  opts.output_tensors.assign(output_tensors.begin(), output_tensors.end());
  opts.get_kernel_time = enable_profiling;
  opts.validate_results.assign(validate_results.begin(),
                               validate_results.end());
  opts.spirv_dump_dir = spirv_dump_dir;

  return opts;
}
