cmake_minimum_required (VERSION 2.6)
project (DeepES)
set(TARGET unit_test_main)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

########## options ##########
# option(WITH_PADDLE "Compile DeepES with PaddleLite framework." OFF)
option(WITH_TORCH "Compile DeepES with Torch framework." OFF)
# message("WITH_PADDLE: "${WITH_PADDLE})
message("WITH_TORCH: "${WITH_TORCH})


find_package(GTest REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# if (WITH_TORCH)
#     list(APPEND CMAKE_PREFIX_PATH "../libtorch")
#     find_package(Torch REQUIRED ON)
#     include_directories("../include/torch")
#     file(GLOB lib_src "../src/torch/*.cc")
# endif()

# file(GLOB demo_src "../demo/*.cc" "../demo/*.cpp")
file(GLOB core_src "../src/*.cc" "../src/*.cpp" "../benchmark/*.cc")
file(GLOB test_src "../test/src/*.cc")

# include_directories("../demo")
include_directories("../include")
include_directories("../benchmark")
include_directories("../test/include")
# link_directories("/usr/lib/x86_64-linux-gnu/")

add_executable(${TARGET} "unit_test.cc" ${core_src} ${test_src} ${lib_src}) #  ${demo_src} 
target_link_libraries(${TARGET} gflags protobuf pthread glog gtest "${TORCH_LIBRARIES}")
