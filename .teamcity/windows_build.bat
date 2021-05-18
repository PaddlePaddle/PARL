@ECHO OFF
rem Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.

rem =================================================
rem       PARL CI Task On Windows Platform
rem =================================================

@ECHO ON
setlocal

rem ------initialize common variable------
set REPO_ROOT=%cd%

set RED="\033[0;31m"
set BLUE="\033[0;34m"
set BOLD="\033[1m"
set NONE="\033[0m"

rem ------ environment variables of cmake and ctest
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

rem ------paddle dygraph unittest
rem ------pre install python requirement----------
call :clean_env

cd %REPO_ROOT%
call conda env remove --name parl_paddle_dygraph_unittest
rmdir "C:\ProgramData\Miniconda3\envs\parl_paddle_dygraph_unittest" /s/q
call echo y | conda create -n parl_paddle_dygraph_unittest python=3.8.5 pip=20.2.1 --no-default-packages
call conda activate parl_paddle_dygraph_unittest || goto conda_error


where python
where pip

pip install -U .
pip install -r .teamcity\windows_requirements_fluid.txt
if %ERRORLEVEL% NEQ 0 (
    goto pip_error
)


call :run_paddle_fluid_test || goto unittest_error
rem ----------------------------------------------


rem ------basic unittest
rem ------ test in python 3.7 and 3.8 environments
for %%v in (3.7 3.8) do (
    rem ------pre install python requirement----------
    call :clean_env
    cd %REPO_ROOT%
    call conda env remove --name parl_unittest_py%%v
    rmdir "C:\ProgramData\Miniconda3\envs\parl_unittest_py"%%v /s/q
    call echo y | conda create -n parl_unittest_py%%v python=%%v pip=20.2.1 --no-default-packages
    call conda activate parl_unittest_py%%v || goto conda_error

    where python
    where pip

    pip install -U .
    pip install -r .teamcity\windows_requirements_paddle.txt
    if %ERRORLEVEL% NEQ 0 (
      goto pip_error
    )

    call xparl stop

    rem ------run parallel unittests
    set IS_TESTING_SERIALLY=OFF
    set IS_TESTING_REMOTE=OFF
    call :run_test_with_cpu || goto unittest_error

    rem uninstall paddle when testing remote module
    pip uninstall -y paddlepaddle

    rem ------run serial unittests
    set IS_TESTING_SERIALLY=ON
    set IS_TESTING_REMOTE=OFF
    call :run_test_with_cpu || goto unittest_error

    rem ------run remote unittests
    set IS_TESTING_SERIALLY=OFF
    set IS_TESTING_REMOTE=ON
    call :run_test_with_cpu || goto unittest_error

)
rem ----------------------------------------------


rem ------import unittest
rem ------pre install python requirement----------
cd %REPO_ROOT%
call :clean_env
call conda env remove --name parl_import_unittest
rmdir "C:\ProgramData\Miniconda3\envs\parl_import_unittest" /s/q
call echo y | conda create -n parl_import_unittest python=3.8.5 pip=20.2.1 --no-default-packages
call conda activate parl_import_unittest || goto conda_error

where python
where pip

pip install -U .
if %ERRORLEVEL% NEQ 0 (
    goto pip_error
)

call :run_import_test || goto unittest_error
rem ----------------------------------------------


rem ------all unittests are successful
goto :success


rem ------functions
rem ------------------------------------------------
:run_test_with_cpu
echo    ===========================================================
echo    run_test_with_cpu IS_TESTING_SERIALLY=%IS_TESTING_SERIALLY%
echo    ===========================================================

if exist %REPO_ROOT%\build (
    rmdir %REPO_ROOT%\build /s/q
)
mkdir %REPO_ROOT%\build
cd %REPO_ROOT%\build


if "%IS_TESTING_SERIALLY%"=="ON" (
	cmake .. -DIS_TESTING_SERIALLY=%IS_TESTING_SERIALLY%
) else if "%IS_TESTING_REMOTE%"=="ON" (
	cmake .. -DIS_TESTING_REMOTE=%IS_TESTING_REMOTE%
) else (
	cmake ..
)

if %ERRORLEVEL% NEQ 0 (
    goto cmake_error
)

if "%IS_TESTING_SERIALLY%"=="ON" (
    ctest -C Release --output-on-failure
) else (
    ctest -C Release --output-on-failure -j5
)
goto:eof
rem ------------------------------------------------


rem ------------------------------------------------
:run_import_test
echo    ===========================================================
echo    run_import_test
echo    ===========================================================


if exist %REPO_ROOT%\build (
    rmdir %REPO_ROOT%\build /s/q
)
mkdir %REPO_ROOT%\build
cd %REPO_ROOT%\build

cmake .. -DIS_TESTING_IMPORT=ON
if %ERRORLEVEL% NEQ 0 (
    goto cmake_error
)

ctest -C Release --output-on-failure
goto:eof
rem ------------------------------------------------


rem ------------------------------------------------
:run_paddle_fluid_test
echo    ===========================================================
echo    run_paddle_fluid_test
echo    ===========================================================

if exist %REPO_ROOT%\build (
    rmdir %REPO_ROOT%\build /s/q
)
mkdir %REPO_ROOT%\build
cd %REPO_ROOT%\build


cmake .. -DIS_TESTING_FLUID=ON
if %ERRORLEVEL% NEQ 0 (
    goto cmake_error
)

ctest -C Release --output-on-failure
goto:eof
rem ------------------------------------------------

rem ------------------------------------------------
:clean_env
echo    ========================================
echo    Clean up environment!
echo    ========================================

taskkill /f /im cmake.exe 2>NUL
taskkill /f /im msbuild.exe 2>NUL
taskkill /f /im git.exe 2>NUL
taskkill /f /im python.exe 2>NUL
taskkill /f /im pip.exe 2>NUL
taskkill /f /im conda.exe 2>NUL
goto:eof
rem ------------------------------------------------


rem ------------------------------------------------
:pip_error
echo    ========================================
echo    pip install failed!
echo    ========================================
exit /b 7
rem ------------------------------------------------

rem ------------------------------------------------
:conda_error
echo    ========================================
echo    conda activate failed!
echo    ========================================
exit /b 7
rem ------------------------------------------------


rem ------------------------------------------------
:cmake_error
echo    ========================================
echo    cmake failed!
echo    ========================================
exit /b 7
rem ------------------------------------------------

rem ------------------------------------------------
:unittest_error
echo    ========================================
echo    Windows CI run failed!
echo    ========================================
exit /b 7
rem ------------------------------------------------


rem ------------------------------------------------
:success
echo    ========================================
echo    Windows CI run successfully!
echo    ========================================
exit /b 0
rem ------------------------------------------------


ENDLOCAL
