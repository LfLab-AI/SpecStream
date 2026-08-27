#include <cstdio>

#include <iostream>

#include <cuda_runtime.h>

#include "check_cuda.h"
#include "libsmctrl.h"

__global__ void echo_sm(int *used_sm, bool echo) {
  if (threadIdx.x != 1)
    return;
  int smIdx;
  asm("mov.u32 %0, %%smid;"
      : "=r"(smIdx));
  if (echo) {
    printf("%d, ", smIdx);
  }
  used_sm[smIdx] = 1;
}

int libsmctrl_validate_stream_mask(void *stream_ptr, int low, int high, bool echo) {
  if (echo) {
    std::cout << "validating stream '" << stream_ptr << "' with mask ranged (" << low << ", " << high << ")\n";
  }
  int *used_sm;
  int num_sms;
  int cuda_device;
  uint32_t num_tpcs;
  int ret_code = 0;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  checkCuda(cudaGetDevice(&cuda_device));
  checkCuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount,
                                   cuda_device));
  if (libsmctrl_get_tpc_info_cuda(&num_tpcs, cuda_device) != 0 ||
      num_tpcs == 0 || num_sms % num_tpcs != 0) {
    std::cerr << "unable to derive the device SM-to-TPC mapping\n";
    return -1;
  }
  int sms_per_tpc = num_sms / num_tpcs;
  checkCuda(cudaMallocManaged(&used_sm, sizeof(int) * num_sms));
  checkCuda(cudaMemset(used_sm, 0, sizeof(int) * num_sms));
  echo_sm<<<256, 12, 0, stream>>>(used_sm, echo);
  checkCuda(cudaStreamSynchronize(stream));
  for (int i = 0; i < num_sms; ++i) {
    if (used_sm[i] == 1) {
      if ((i / sms_per_tpc < low) or (i / sms_per_tpc >= high)) {
        std::cout << "SM " << i << " shouldn't be used\n";
        ret_code = -1;
      }
    } else if ((low <= i / sms_per_tpc) && (i / sms_per_tpc < high)) {
      std::cout << "SM " << i << " should be used\n";
      ret_code = -1;
    }
  }
  checkCuda(cudaFree(used_sm));
  return ret_code;
}
