cmake_minimum_required(VERSION 3.20)
project(po_sbr_optix LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Set OPTIX_ROOT to your OptiX SDK install path, e.g.
# cmake -S c_optix -B c_optix/build -DOPTIX_ROOT=/opt/NVIDIA-OptiX-SDK-7.7.0
if (NOT OPTIX_ROOT)
  message(FATAL_ERROR "OPTIX_ROOT is not set. Point it to OptiX SDK root.")
endif()

add_library(po_optix STATIC optix_solver.cu)
target_include_directories(po_optix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPTIX_ROOT}/include)
target_link_libraries(po_optix PUBLIC CUDA::cudart)

add_executable(po_rcs_optix main_rcs_optix.cpp)
target_link_libraries(po_rcs_optix PRIVATE po_optix)

find_library(OPTIX_LIB optix.1 PATHS ${OPTIX_ROOT}/lib64 ${OPTIX_ROOT}/lib NO_DEFAULT_PATH)
if (NOT OPTIX_LIB)
  message(FATAL_ERROR "Could not find optix.1 under ${OPTIX_ROOT}")
endif()
target_link_libraries(po_optix PUBLIC ${OPTIX_LIB})
