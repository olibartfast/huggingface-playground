cmake_minimum_required(VERSION 3.10)
project(ViTPoseInference)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "/home/oli/dependencies/onnxruntime-linux-x64-gpu-1.19.2")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(vitpose_inference main.cpp)

target_link_libraries(vitpose_inference 
    ${OpenCV_LIBS} 
    onnxruntime
)

# Set RPATH so it finds the onnxruntime shared library at runtime
set_target_properties(vitpose_inference PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
