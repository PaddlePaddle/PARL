#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0)

enable_testing()

option(WITH_TESTING "Include unit testing" ON)
option(IS_TESTING_SERIALLY "testing scripts that cannot run in parallel" OFF)
option(IS_TESTING_IMPORT "testing import parl" OFF)
option(IS_TESTING_DOCS "testing compling the docs" OFF)
option(IS_TESTING_GPU "testing GPU environment" OFF)
option(IS_TESTING_TORCH "testing torch parts" OFF)

set(PADDLE_PYTHON_PATH "" CACHE STRING "Python path to PaddlePaddle Fluid")

function(py_test TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${TARGET_NAME}
        COMMAND python -u ${py_test_SRCS} ${py_test_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 300)
endfunction()

function(import_test TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_test(NAME ${TARGET_NAME}_with_empty_env
        COMMAND /root/miniconda3/envs/empty_env/bin/python -u ${py_test_SRCS} ${py_test_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

endfunction()

function(docs_test)
    add_test(NAME compling_docs_test
        COMMAND bash ./docs/test.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

if (WITH_TESTING)
    if (IS_TESTING_IMPORT)
        set(src "parl/tests/import_test")
        import_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
    elseif (IS_TESTING_DOCS)
        docs_test()
    elseif (IS_TESTING_SERIALLY)
        file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test_alone.py")
        string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
        foreach(src ${TEST_OPS})
            py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
        endforeach()
    elseif (IS_TESTING_TORCH)
      file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test_torch.py")
      string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
      foreach(src ${TEST_OPS})
          py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
      endforeach()
    else ()
        file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test.py")
        string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
        foreach(src ${TEST_OPS})
          if (${src} MATCHES ".*remote.*")
              if (NOT IS_TESTING_GPU)
                  py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
              endif()
          else()
              py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
          endif()
        endforeach()
    endif()
endif()
