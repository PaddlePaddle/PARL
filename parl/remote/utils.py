#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import cloudpickle
import sys
import inspect
from contextlib import contextmanager
import os
import pkg_resources
from parl.utils import logger

__all__ = [
    'redirect_output_to_file', 'get_subfiles_recursively', 'get_version',
    'XPARL_PYTHON'
]


@contextmanager
def redirect_output_to_file(stdout_file_path, stderr_file_path):
    """Redirect stdout (e.g., `print`) and stderr (e.g., `warning/error`) to given files respectively.

    Args:
        stdout_file_path: Path of the file to output the stdout.
        stderr_file_path: Path of the file to output the stderr.

    Example:
    >>> print('test')
    test
    >>> with redirect_output_to_file('stdout.log', 'stderr.log'):
    ...     print('test')  # Output nothing, `test` is printed to `stdout.log`.
    >>> print('test')
    test
    """

    origin_stdout = sys.stdout
    origin_stderr = sys.stderr

    stdout_f = open(stdout_file_path, 'a')
    stderr_f = open(stderr_file_path, 'a')

    sys.stdout = stdout_f
    sys.stderr = stderr_f

    # NOTE: we should add the handler after executing the above code.
    handler = logger.add_stdout_handler()

    try:
        yield
    finally:
        sys.stdout = origin_stdout
        sys.stderr = origin_stderr

        stdout_f.close()
        stderr_f.close()

        logger.remove_handler(handler)


def get_subfiles_recursively(folder_path):
    '''
    Get subfiles under 'folder_path' recursively
    Args:
        folder_path: A folder(dir) whose subfiles/subfolders will be returned.

    Returns:
        python_files: A list including subfiles endwith '.py'.
        other_files: A list including subfiles not endwith '.py'.
        empty_subfolders: A list including empty subfolders.
    '''
    if not os.path.exists(folder_path):
        raise ValueError("Path '{}' don't exist.".format(folder_path))
    elif not os.path.isdir(folder_path):
        raise ValueError('Input should be a folder, not a file.')
    else:
        python_files = []
        other_files = []
        empty_subfolders = []
        for root, dirs, files in os.walk(folder_path):
            if files:
                for sub_file in files:
                    if sub_file.endswith('.py'):
                        python_files.append(
                            os.path.normpath(os.path.join(root, sub_file)))
                    else:
                        other_files.append(
                            os.path.normpath(os.path.join(root, sub_file)))
            elif len(dirs) == 0:
                empty_subfolders.append(os.path.normpath(root))
        return python_files, other_files, empty_subfolders


def get_version(module_name):
    ''' Check if the python environment has installed the module or package.
        return the version of the module if the module is installed,
        return None otherwise.
    Args:
        module_name (str): module to be checked
    Returns:
        has_module: str (if the module is installed) or None
    '''
    assert isinstance(module_name, str), '"module_name" should be a string!'
    try:
        __import__(module_name)
    except ImportError:
        return None
    else:
        module_version = pkg_resources.get_distribution(module_name).version
        return module_version


def get_xparl_python():
    """Users can specify the executable python of xparl command
    by exporting XPARL_PYTHON environment variable

    Examples:
    ```bash
    export XPARL_PYTHON='/opt/compiler/gcc-10/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-10/lib:~/miniconda3/envs/py36/lib: ~/miniconda3/envs/py36/bin/python'
    xparl start --port 8010
    ```

    Returns:
        xparl_python (list): executable python of xparl command
    """
    xparl_python = sys.executable
    env_name = "XPARL_PYTHON"
    if env_name in os.environ and os.environ[env_name]:
        assert isinstance(os.environ[env_name], str)
        xparl_python = os.environ[env_name]
    return xparl_python.split()


XPARL_PYTHON = get_xparl_python()
