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
from parl.utils import isnotebook

__all__ = [
    'load_remote_class', 'redirect_stdout_to_file', 'locate_remote_file'
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


def load_remote_class(file_name, class_name, end_of_file):
    """
  load a class given its file_name and class_name.

  Args:
    file_name: specify the file to load the class
    class_name: specify the class to be loaded
    end_of_file: line ID to indicate the last line that defines the class.

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
    module_name = module_name.lstrip('.' + os.sep).replace(os.sep, '.')
    mod = __import__(module_name, globals(), locals(), [class_name], 0)
    cls = getattr(mod, class_name)
    return cls


@contextmanager
def redirect_stdout_to_file(file_path):
    """Redirect stdout (e.g., `print`) to specified file.

    Args:
        file_path: Path of the file to output the stdout.

    Example:
    >>> print('test')
    test
    >>> with redirect_stdout_to_file('test.log'):
    ...     print('test')  # Output nothing, `test` is printed to `test.log`.
    >>> print('test')
    test
    """
    tmp = sys.stdout
    f = open(file_path, 'a')
    sys.stdout = f
    try:
        yield
    finally:
        sys.stdout = tmp
        f.close()


def locate_remote_file(module_path):
    """xparl has to locate the file that has the class decorated by parl.remote_class. 
    This function returns the relative path between this file and the entry file.
    Note that this function should support the jupyter-notebook environment.

    Args:
        module_path: Absolute path of the module.

    Example:
        module_path: /home/user/dir/subdir/my_module
        entry_file: /home/user/dir/main.py
        --------> relative_path: subdir/my_module
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
    if entry_path is None or \
        (module_path.startswith(os.sep) and entry_path != module_path[:len(entry_path)]):
        raise FileNotFoundError("cannot locate the remote file")
    if module_path.startswith(os.sep):
        relative_module_path = '.' + module_path[len(entry_path):]
    else:
        relative_module_path = module_path
    return relative_module_path
