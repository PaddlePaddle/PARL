cmake_minimum_required (VERSION 2.6)
project (EvoKit_demo)
set(TARGET parallel_main)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

list(APPEND CMAKE_PREFIX_PATH "./libtorch")
find_package(Torch REQUIRED ON)
set(demo "${PROJECT_SOURCE_DIR}/cartpole_solver_parallel.cc")


########## main ##########
add_executable(${TARGET} ${demo} ${framework_src}) 
target_link_libraries(${TARGET} gflags protobuf pthread glog)

########## Torch libraries ##########
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")


########## EvoKit libraries ##########
list(APPEND CMAKE_PREFIX_PATH "./libevokit/cmake/Torch")
find_package(EvoKit)
target_link_libraries(${TARGET} "${EVOKIT_LIBRARY}")
