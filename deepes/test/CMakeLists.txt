cmake_minimum_required (VERSION 2.6)
project (DeepES)
set(TARGET unit_test_main)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTest REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Torch lib
list(APPEND CMAKE_PREFIX_PATH "../libtorch")
find_package(Torch REQUIRED ON)

# include and source
file(GLOB test_src "../test/src/*.cc")
file(GLOB core_src "../core/src/*.cc")
file(GLOB agent_src "../torch/src/*.cc")

include_directories("../torch/include")
include_directories("../core/include")
include_directories("../benchmark")
include_directories("../test/include")


add_executable(${TARGET} "unit_test.cc" ${core_src} ${agent_src} ${test_src})
target_link_libraries(${TARGET} gflags protobuf pthread glog gtest "${TORCH_LIBRARIES}")
