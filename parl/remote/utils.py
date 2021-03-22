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

import sys
from contextlib import contextmanager
import os
import pkg_resources
from parl.utils import isnotebook, logger, format_uniform_path

__all__ = [
    'load_remote_class', 'redirect_output_to_file', 'locate_remote_file',
    'get_subfiles_recursively', 'get_version'
]


def simplify_code(code, end_of_file):
    """
  @parl.remote_actor has to use this function to simplify the code.
  To create a remote object, PARL has to import the module that contains the decorated class.
  It may run some unnecessary code when importing the module, and this is why we use this function
  to simplify the code.

  For example.
  @parl.remote_actor
  class A(object):
    def add(self, a, b):
    return a + b
  def data_process():
    XXXX
  ------------------>
  The last two lines of the above code block will be removed as they are not class-related.
  """
    to_write_lines = []
    for i, line in enumerate(code):
        if line.startswith('parl.connect'):
            continue
        if i < end_of_file - 1:
            to_write_lines.append(line)
        else:
            break
    return to_write_lines


def load_remote_class(file_name, class_name, end_of_file, in_sys_path):
    """
  load a class given its file_name and class_name.

  Args:
    file_name: specify the file to load the class
    class_name: specify the class to be loaded
    end_of_file: line ID to indicate the last line that defines the class.
    in_sys_path: whether the path of the remote class is in the environment path (sys.path).

  Return:
    cls: the class to load
  """
    with open(file_name + '.py') as t_file:
        code = t_file.readlines()
    code = simplify_code(code, end_of_file)
    #folder/xx.py -> folder/xparl_xx.py
    file_name = file_name.split(os.sep)
    prefix = os.sep.join(file_name[:-1])
    if prefix == "":
        prefix = '.'
    module_name = prefix + os.sep + 'xparl_' + file_name[-1]
    tmp_file_name = module_name + '.py'
    with open(tmp_file_name, 'w') as t_file:
        for line in code:
            t_file.write(line)

    if in_sys_path:
        new_file_name = "xparl_" + file_name[-1]
        # the path of the remote class is in the sys.path, we can import it directly.
        mod = __import__(new_file_name)
    else:
        module_name = module_name.lstrip('.' + os.sep).replace(os.sep, '.')
        mod = __import__(module_name, globals(), locals(), [class_name], 0)

    cls = getattr(mod, class_name)

    os.remove(tmp_file_name)
    return cls


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


def locate_remote_file(module_path):
    """xparl has to locate the file that has the class decorated by parl.remote_class. 

    If the entry_path is the prefix of the absolute path of the module, which means the 
    file of the module is in the directory of entry or its subdirectories, this function
    will return the relative path between this file and the entry file.

    Otherwise, it means that the user accesses the module by adding the path of the
    module to the environment path (sys.path), this function will return the module_path
    directly and the remote job also need access the module by adding the path of the 
    module to the sys.path.

    Note that this function should support the jupyter-notebook environment.

    Args:
        module_path: Absolute path of the module, where the class decorated by
                     parl.remote_class is located.
    
    Returns:
        (return_path, in_sys_path)

        return_path(str): relative path of the module, if the entry_path is the prefix of the absolute
                          path of the module, else module_path.

        in_sys_path(bool): False, if the entry_path is the prefix of the absolute path of the module,
                           else True.
    Example:
        module_path: /home/user/dir/subdir/my_module (or) ./dir/main
        entry_file: /home/user/dir/main.py
        --------> return_path: subdir/my_module

        module_path: /home/user/other_dir/subdir/my_module
        entry_file: /home/user/dir/main.py
        --------> return_path: /home/user/other_dir/subdir/my_module

        module_path: ../other_dir/subdir/my_module
        entry_file: /home/user/dir/main.py
        --------> return_path: ../other_dir/subdir/my_module
    """
    if isnotebook():
        entry_path = os.getcwd()
    else:
        entry_file = sys.argv[0]
        entry_file = entry_file.split(os.sep)[-1]
        entry_path = None
        for path in sys.path:
            to_check_path = os.path.join(path, entry_file)
            if os.path.isfile(to_check_path):
                entry_path = path
                break

    if entry_path is None:
        raise FileNotFoundError("cannot locate the remote file")

    # fix pycharm issue: https://github.com/PaddlePaddle/PARL/issues/350
    module_path = format_uniform_path(module_path)
    entry_path = format_uniform_path(entry_path)

    # transfer the relative path to the absolute path
    abs_module_path = module_path
    if not os.path.isabs(abs_module_path):
        abs_module_path = os.path.abspath(abs_module_path)

    if os.sep in abs_module_path \
            and entry_path != abs_module_path[:len(entry_path)]:
        # the file of the module is not in the directory of entry or its subdirectories
        return module_path, True

    if os.sep in abs_module_path:
        relative_module_path = '.' + abs_module_path[len(entry_path):]
    else:
        relative_module_path = abs_module_path

    return relative_module_path, False


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
