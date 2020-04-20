# FindEvoKit
# -------
#
# Finds the EvoKit library
#
#   EvoKit

include(FindPackageHandleStandardArgs)

if (DEFINED ENV{EVOKIT_INSTALL_PREFIX})
  set(EVOKIT_INSTALL_PREFIX $ENV{EVOKIT_INSTALL_PREFIX})
else()
  # Assume we are in <install-prefix>/cmake/Torch/EvoKitConfig.cmake
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(EVOKIT_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
endif()

# Include directories.
if (EXISTS "${EVOKIT_INSTALL_PREFIX}/include")
  set(EVOKIT_INCLUDE_DIRS
    ${EVOKIT_INSTALL_PREFIX}/include
    ${EVOKIT_INSTALL_PREFIX}/torch)
else()
  set(EVOKIT_INCLUDE_DIRS
    ${EVOKIT_INSTALL_PREFIX}/include
    ${EVOKIT_INSTALL_PREFIX}/torch)
endif()

message(STATUS "EVOKIT_INSTALL_PREFIX: ${EVOKIT_INSTALL_PREFIX}")

find_library(EVOKIT_LIBRARY libEvoKit_torch.a PATHS "${EVOKIT_INSTALL_PREFIX}/lib")
include_directories("${EVOKIT_INSTALL_PREFIX}/torch")
include_directories("${EVOKIT_INSTALL_PREFIX}/include")
find_package_handle_standard_args(EvoKit DEFAULT_MSG EVOKIT_LIBRARY EVOKIT_INCLUDE_DIRS)
message(STATUS "EVOKIT_FOUND: ${EVOKIT_FOUND}")
