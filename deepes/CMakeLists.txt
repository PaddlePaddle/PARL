cmake_minimum_required (VERSION 2.6)
project (DeepES)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Torch REQUIRED ON)

file(GLOB demo_src "demo/*.cpp")
file(GLOB core_src "src/*.cpp")
file(GLOB pb_src "src/*.cc")

include_directories("include")
include_directories("demo")
include_directories("benchmark")
link_directories("/usr/lib/x86_64-linux-gnu/")

add_executable(parallel_main "./demo/cartpole_solver_parallel.cpp" ${core_src} ${pb_src} ${benchmark_src})
target_link_libraries(parallel_main gflags protobuf pthread glog "${TORCH_LIBRARIES}")
