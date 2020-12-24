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


rem ------pre install python requirement----------
conda activate parl_unittest_py38
where python
where pip

pip install .
pip install -r .teamcity\windows_requirements.txt

rem ------run parallel unitests
set IS_TESTING_SERIALLY=OFF
call :run_test_with_cpu || goto unittest_error

rem ------run serial unitests
set IS_TESTING_SERIALLY=ON
call :run_test_with_cpu || goto unittest_error

goto :success

rem ------------------------------------------------
:run_test_with_cpu
echo    ===========================================================
echo    run_test_with_cpu IS_TESTING_SERIALLY=%IS_TESTING_SERIALLY%
echo    ===========================================================

if not exist build (
    mkdir %REPO_ROOT%\build
)
cd %REPO_ROOT%\build

cmake .. -DIS_TESTING_SERIALLY=%IS_TESTING_SERIALLY%
if "%IS_TESTING_SERIALLY%"=="ON" (
    ctest --output-on-failure
) else (
    ctest --output-on-failure -j20 --verbose 
)

cd ${REPO_ROOT}
rm -rf ${REPO_ROOT}\build
goto:eof

rem ------------------------------------------------
:unittest_error
echo    ========================================
echo    Windows CI run failed!
echo    ========================================
exit /b 7

rem ------------------------------------------------
:success
echo    ========================================
echo    Windows CI run successfully!
echo    ========================================
exit /b 0

ENDLOCAL
