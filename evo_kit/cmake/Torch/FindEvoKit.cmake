message(STATUS "WITH_PADDLE: ${WITH_PADDLE}")
message(STATUS "WITH_TORCH: ${WITH_TORCH}")

set(THE_LIB_PATH ${PROJECT_SOURCE_DIR}/libevokit)
if (WITH_PADDLE)
  set(LIB_FILE "libEvoKit_paddle.a")
elseif (WITH_TORCH)
  set(LIB_FILE "libEvoKit_torch.a")
else ()
  message("ERROR: please use -DWITH_TORCH=ON or -DWITH_PADDLE=ON while cmake.")
endif()

find_path(EVOKIT_INCLUDE_DIR sampling_method.h ${THE_LIB_PATH}/include/evo_kit)
find_library(EVOKIT_LIBRARY ${LIB_FILE} ${THE_LIB_PATH}/lib)

if (EVOKIT_INCLUDE_DIR AND EVOKIT_LIBRARY)
  set(EVOKIT_FOUND TRUE)
endif(EVOKIT_INCLUDE_DIR AND EVOKIT_LIBRARY)

if(EVOKIT_FOUND)
  if(NOT EVOKIT_FIND_QUIETLY)
  message(STATUS "Found EVOKIT: ${EVOKIT_LIBRARY}")
  endif(NOT EVOKIT_FIND_QUIETLY)
endif(EVOKIT_FOUND)