cmake_minimum_required(VERSION 3.17)
project(SpecStreamSMCtrl LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_library(smctrl SHARED
  src/libsmctrl_core.c
  src/libsmctrl_validator.cu
)
target_include_directories(smctrl PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(smctrl PRIVATE CUDA::cuda_driver CUDA::cudart)
set_target_properties(smctrl PROPERTIES
  PUBLIC_HEADER src/libsmctrl.h
  OUTPUT_NAME smctrl
)

option(SPECSTREAM_SMCTRL_BUILD_TESTS "Build the stream-mask validator" ON)
if(SPECSTREAM_SMCTRL_BUILD_TESTS)
  add_executable(specstream_smctrl_validator test/test_validator.cu)
  target_include_directories(specstream_smctrl_validator PRIVATE src)
  target_link_libraries(specstream_smctrl_validator PRIVATE smctrl CUDA::cudart)
endif()
