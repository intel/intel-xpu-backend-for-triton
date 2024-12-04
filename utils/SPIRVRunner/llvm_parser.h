#ifndef LLVM_PARSER_H
#define LLVM_PARSER_H

#include "llvm/Support/CommandLine.h"

class command_line_parser {
public:
  struct options {
    std::string output_tensor;
    bool enable_profiling;
  };

  command_line_parser(int argc, char **argv);
  options parse();

private:
  int argc;
  char **argv;
};
#endif
