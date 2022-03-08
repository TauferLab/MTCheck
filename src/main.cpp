#include <iostream>
#include "dedup.hpp"
#include <string>
#include <vector>

int main(int argc, char**argv) {
  std::string full_chkpt(argv[1]);
  std::string incr_chkpt = full_chkpt + ".incr_chkpt";

  std::vector<std::string> prev_chkpt;

  for(int i=2; i<argc; i++) {
    prev_chkpt.push_back(std::string(argv[i]));
  }

  deduplicate_module_t module;

  module.deduplicate_file(full_chkpt, incr_chkpt, prev_chkpt);
}

