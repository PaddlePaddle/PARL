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
option(IS_TESTING_TORCH "testing torch parts" OFF)
option(IS_TESTING_FLUID "testing fluid parts" OFF)
option(IS_TESTING_REMOTE "testing remote module" OFF)

set(PADDLE_PYTHON_PATH "" CACHE STRING "Python path to PaddlePaddle Fluid")

function(py_test TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if (${FILE_NAME} MATCHES ".*abs_test.py")
        add_test(NAME ${TARGET_NAME}"_with_abs_path"
            COMMAND python -u ${py_test_SRCS} ${py_test_ARGS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(${TARGET_NAME}"_with_abs_path" PROPERTIES TIMEOUT 800)
    else()
        get_filename_component(WORKING_DIR ${py_test_SRCS} DIRECTORY)
        get_filename_component(FILE_NAME ${py_test_SRCS} NAME)
        get_filename_component(COMBINED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${WORKING_DIR} ABSOLUTE)
        add_test(NAME ${TARGET_NAME}
            COMMAND python -u ${FILE_NAME} ${py_test_ARGS}
            WORKING_DIRECTORY ${COMBINED_PATH})
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 800)
    endif()
endfunction()

function(import_test TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_test(NAME ${TARGET_NAME}_with_empty_env
        COMMAND python -u ${py_test_SRCS} ${py_test_ARGS}
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
    elseif (IS_TESTING_FLUID)
        file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test_fluid.py")
        string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
        foreach(src ${TEST_OPS})
            py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
        endforeach()
    elseif (IS_TESTING_REMOTE)
        file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test.py")
        string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
        foreach(src ${TEST_OPS})
            if (${src} MATCHES ".*remote.*")
                py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
            endif()
        endforeach()
    else ()
        file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test.py")
        string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
        foreach(src ${TEST_OPS})
            if (NOT ${src} MATCHES ".*remote.*")
                py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
            endif()
        endforeach()
        file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test_paddle.py")
        string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
        foreach(src ${TEST_OPS})
            py_test(${src} SRCS ${src}.py ENVS ${PADDLE_PYTHON_PATH})
        endforeach()
    endif()
endif()
