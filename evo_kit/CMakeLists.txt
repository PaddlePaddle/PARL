cmake_minimum_required (VERSION 2.6)
project (EvoKit)
set(TARGET parallel_main)

########## options ##########
option(WITH_PADDLE "Compile EvoKit with PaddleLite framework." OFF)
option(WITH_TORCH "Compile EvoKit with Torch framework." OFF)

message("WITH_PADDLE: "${WITH_PADDLE})
message("WITH_TORCH: "${WITH_TORCH})

if (NOT (WITH_PADDLE OR WITH_TORCH))
  message("ERROR: You should choose at least one framework to compile EvoKit.")
  return()
elseif(WITH_PADDLE AND WITH_TORCH)
  message("ERROR: You cannot choose more than one framework to compile EvoKit.")
  return()
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


file(GLOB src "core/src/*.cc" "core/proto/evo_kit/*.cc")
include_directories("core/include")
include_directories("core/proto")
include_directories("benchmark")

########## PaddleLite config ##########
if (WITH_PADDLE)
  add_definitions(-g -O3 -pthread)

  include_directories("paddle/include")
  include_directories("${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/include" 
                   "${PROJECT_SOURCE_DIR}/inference_lite_lib/third_party/mklml/include")
  link_directories("${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/lib" 
                   "${PROJECT_SOURCE_DIR}/inference_lite_lib/third_party/mklml/lib")
  
  file(GLOB framework_src "paddle/src/*.cc")
  set(demo "${PROJECT_SOURCE_DIR}/demo/paddle/cartpole_solver_parallel.cc")
  #set(demo "${PROJECT_SOURCE_DIR}/demo/paddle/cartpole_async_solver.cc")
########## Torch config ##########
elseif (WITH_TORCH)
  list(APPEND CMAKE_PREFIX_PATH "./libtorch")
  find_package(Torch REQUIRED ON)

  include_directories("torch/include")
  include_directories("demo/torch")

  file(GLOB framework_src "torch/src/*.cc")
  set(demo "${PROJECT_SOURCE_DIR}/demo/torch/cartpole_solver_parallel.cc")
else ()
  message("ERROR: You should choose at least one framework to compile EvoKit.")
endif()

add_executable(${TARGET} ${demo} ${src} ${framework_src}) 

target_link_libraries(${TARGET} gflags protobuf pthread glog)

########## PaddleLite libraries ##########
if (WITH_PADDLE)
  target_link_libraries(${TARGET} -lpaddle_full_api_shared)
  target_link_libraries(${TARGET} -lmklml_intel)
  target_link_libraries(${TARGET} -ldl)
########## Torch libraries ##########
elseif (WITH_TORCH)
  target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")
endif()
