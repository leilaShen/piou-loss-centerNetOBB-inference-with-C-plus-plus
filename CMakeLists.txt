cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

#set(Torch_DIR /media/vision/Data/software/pytorch_1.7.1/pytorch/torch/share/cmake/Torch) 
set(Torch_DIR /usr/local/libtorch/share/cmake/Torch) 
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(example-app main.cpp PostProcess.cpp)
#target_link_libraries(example-app "${TORCH_LIBRARIES}" "/home/dell/shenlei/piou_inference/build/libdcn_v2_cuda_forward_v2.so" "${OpenCV_LIBS}")
target_link_libraries(example-app "${TORCH_LIBRARIES}" "/media/vision/Data/work/piou_inference/lib/libdcn_v2_cuda_forward_v2.so" "${OpenCV_LIBS}")
#link_libraries("/home/dell/shenlei/torch_inferer/test_use_libtorch/lib/libdcn_v2_cuda_forward_v2.so")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
