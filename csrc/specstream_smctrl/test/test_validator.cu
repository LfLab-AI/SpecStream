#include <cstring>
#include <iostream>

#include "libsmctrl.h"
#include "check_cuda.h"

using namespace std;

int main(int argc, char **argv){
  if (argc != 3 && argc != 4) {
    cerr << "usage: " << argv[0]
         << " <tpc-low> <tpc-high-exclusive> [stream|global]\n";
    return 2;
  }
  int low = atoi(argv[1]);
  int hi = atoi(argv[2]);
  if (low < 0 || hi <= low) {
    cerr << "invalid TPC range\n";
    return 2;
  }
  const char *scope = argc == 4 ? argv[3] : "stream";
  if (strcmp(scope, "stream") != 0 && strcmp(scope, "global") != 0) {
    cerr << "mask scope must be 'stream' or 'global'\n";
    return 2;
  }
  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream));
  if (strcmp(scope, "global") == 0) {
    uint32_t total_tpcs = 0;
    int device = 0;
    checkCuda(cudaGetDevice(&device));
    if (libsmctrl_get_tpc_info_cuda(&total_tpcs, device) != 0 ||
        total_tpcs == 0 || total_tpcs > 64) {
      cerr << "global mask validator requires a device with 1..64 TPCs\n";
      return 1;
    }
    uint64_t mask;
    if (libsmctrl_make_mask(&mask, low, hi) != 0) {
      cerr << "failed to create global mask\n";
      return 1;
    }
    cout << "using process-global QMD/TMD mask backend\n";
    libsmctrl_set_global_mask(mask);
  } else {
    uint128_t mask;
    if (libsmctrl_make_mask_ext(&mask, low, hi) != 0 ||
        libsmctrl_set_stream_mask_ext(stream, mask) != 0) {
      cerr << "failed to install stream mask\n";
      return 1;
    }
  }
  int ret_code = libsmctrl_validate_stream_mask(stream, low, hi, true);
  checkCuda(cudaStreamDestroy(stream));
  if(ret_code == 0){
    cout << "test passed\n";
  } else {
    cout << "test failed\n";
  }
  return ret_code == 0 ? 0 : 1;
}
