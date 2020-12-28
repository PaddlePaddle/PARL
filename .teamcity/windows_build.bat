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
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/


rem ------basic unittest
for %%v in (3.6 3.7 3.8) do (
	rem ------pre install python requirement----------
	set env_name=parl_unittest_py%%v
	call conda env remove --name %env_name%
	call echo y | conda create -n %env_name% python=%%v --no-default-packages
	call conda activate %env_name%

	where python
	where pip

	pip install  -i https://mirror.baidu.com/pypi/simple -U .
	pip install -i https://mirror.baidu.com/pypi/simple -r .teamcity\windows_requirements.txt
	if %ERRORLEVEL% NEQ 0 (
	    goto pip_error
	)

	call xparl stop

	rem ------run parallel unittests
	set IS_TESTING_SERIALLY=OFF
	call :run_test_with_cpu || goto unittest_error

	rem ------run serial unittests
	set IS_TESTING_SERIALLY=ON
	call :run_test_with_cpu || goto unittest_error
)
rem ----------------------------------------------


rem ------import unittest
rem ------pre install python requirement----------
call conda env remove --name parl_import_unittest
call echo y | conda create -n parl_import_unittest python=3.8.5 --no-default-packages
call conda activate parl_import_unittest

where python
where pip

pip install  -i https://mirror.baidu.com/pypi/simple -U .
pip install -i https://mirror.baidu.com/pypi/simple -r .teamcity\windows_requirements.txt
if %ERRORLEVEL% NEQ 0 (
		goto pip_error
)

call :run_import_test || goto unittest_error
rem ----------------------------------------------


rem ------paddle dygraph unittest
rem ------pre install python requirement----------
call conda env remove --name parl_paddle_dygraph_unittest
call echo y | conda create -n parl_paddle_dygraph_unittest python=3.8.5 --no-default-packages
call conda activate parl_paddle_dygraph_unittest

where python
where pip

pip install  -i https://mirror.baidu.com/pypi/simple -U .
pip install -i https://mirror.baidu.com/pypi/simple -r .teamcity\windows_requirements_paddle.txt
if %ERRORLEVEL% NEQ 0 (
		goto pip_error
)

call :run_paddle_dygraph_test || goto unittest_error
rem ----------------------------------------------


rem ------all unittests are successful
goto :success







rem ------functions
rem ------------------------------------------------
:run_test_with_cpu
echo    ===========================================================
echo    run_test_with_cpu IS_TESTING_SERIALLY=%IS_TESTING_SERIALLY%
echo    ===========================================================

if not exist %REPO_ROOT%\build (
    mkdir %REPO_ROOT%\build
)
cd %REPO_ROOT%\build

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

cmake .. -DIS_TESTING_SERIALLY=%IS_TESTING_SERIALLY%
if %ERRORLEVEL% NEQ 0 (
		goto cmake_error
)

if "%IS_TESTING_SERIALLY%"=="ON" (
    ctest -C Debug --output-on-failure
) else (
    ctest -C Debug --output-on-failure -j10 --verbose 
)

cd %REPO_ROOT%
rmdir %REPO_ROOT%\build /s/q
goto:eof
rem ------------------------------------------------


rem ------------------------------------------------
:run_import_test
echo    ===========================================================
echo    run_import_test
echo    ===========================================================

if not exist %REPO_ROOT%\build (
    mkdir %REPO_ROOT%\build
)
cd %REPO_ROOT%\build

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

cmake .. -DIS_TESTING_IMPORT=ON
if %ERRORLEVEL% NEQ 0 (
		goto cmake_error
)

ctest -C Debug --output-on-failure

cd %REPO_ROOT%
rmdir %REPO_ROOT%\build /s/q
goto:eof
rem ------------------------------------------------


rem ------------------------------------------------
:run_paddle_dygraph_test
echo    ===========================================================
echo    run_paddle_dygraph_test
echo    ===========================================================

if exist %REPO_ROOT%\build (
    rmdir %REPO_ROOT%\build /s/q
)
mkdir %REPO_ROOT%\build
cd %REPO_ROOT%\build

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

cmake .. -DIS_TESTING_PADDLE=ON
if %ERRORLEVEL% NEQ 0 (
		goto cmake_error
)

ctest -C Debug --output-on-failure

cd %REPO_ROOT%
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
