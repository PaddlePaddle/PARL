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

import os
import cloudpickle
import sys
import inspect
from parl.utils import isnotebook, format_uniform_path

__all__ = ['dump_remote_class', 'load_remote_class']


def simplify_code(code, end_of_file):
    """
  @parl.remote_class has to use this function to simplify the code.
  To create a remote object, PARL has to import the module that contains the decorated class.
  It may run some unnecessary code when importing the module, and this is why we use this function
  to simplify the code.

  For example.
  @parl.remote_class
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
        if line.lstrip().startswith('parl.connect'):
            continue
        if i < end_of_file - 1:
            to_write_lines.append(line)
        else:
            break
    return to_write_lines


def is_implemented_in_notebook(cls):
    """Check if the remote class is implemented in the environments like notebook(e.g., ipython, notebook).

    Args:
        cls: class
    """
    assert inspect.isclass(cls)

    if hasattr(cls, '__module__'):
        cls_module = sys.modules.get(cls.__module__)
        if getattr(cls_module, '__file__', None):
            return False
    return True


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


def dump_remote_class(cls):
    """
    Args:
        cls: class decorated by @parl.remote_class
    """
    in_notebook = is_implemented_in_notebook(cls)

    if in_notebook:
        dumped_class_info = cls
    else:
        module_path = inspect.getfile(cls)
        if module_path.endswith('pyc'):
            module_path = module_path[:-4]
        elif module_path.endswith('py'):
            module_path = module_path[:-3]
        else:
            raise FileNotFoundError(
                "cannot not find the module:{}".format(module_path))

        if ".." in module_path:
            # append relative path (E.g. "../a/") to the sys.path,
            # inspect.getfile may return an abnormal path (E.g. "/home/user/../a/").
            module_path = module_path[module_path.index(".."):]

        res = inspect.getfile(cls)
        file_path, in_sys_path = locate_remote_file(module_path)
        cls_source = inspect.getsourcelines(cls)
        end_of_file = cls_source[1] + len(cls_source[0])
        class_name = cls.__name__

        dumped_class_info = [
            file_path, class_name, end_of_file, in_sys_path, sys.path
        ]

    return cloudpickle.dumps([in_notebook, dumped_class_info])


def load_remote_class(remote_class_info):
    """load a class given related info dumped in the client.

    Args:
      remote_class_info: [in_notebook, dumped_class_info]
                         - in_notebook: whether the remote class is implemented in the environments like notebook (e.g., ipython, notebook).
                         - dumped_class_info: information of dumped remote class (decided by `in_notebook`).

    Return:
      cls: the class to load
    """

    in_notebook, dumped_class_info = cloudpickle.loads(remote_class_info)

    if in_notebook:
        cls = dumped_class_info
    else:
        """
        dumped_class_info:
            file_name: specify the file to load the class
            class_name: specify the class to be loaded
            end_of_file: line ID to indicate the last line that defines the class.
            in_sys_path: whether the path of the remote class is in the environment path (sys.path).
            client_sys_path: the environment paths of the client
        """
        file_name, class_name, end_of_file, in_sys_path, client_sys_path = dumped_class_info

        with open(file_name + '.py') as t_file:
            code = t_file.readlines()
        code = simplify_code(code, end_of_file)
        #folder/xx.py -> folder/xparl_xx.py
        file_name = file_name.split(os.sep)
        prefix = os.sep.join(file_name[:-1])
        if prefix == "":
            prefix = '.'

        # Add pid to fix:
        #    https://github.com/PaddlePaddle/PARL/issues/611
        #    Multiple jobs may write the same file when `in_sys_path` is True.
        new_file_name = 'xparl_{}'.format(os.getpid()) + file_name[-1]
        module_name = prefix + os.sep + new_file_name
        tmp_file_name = module_name + '.py'
        with open(tmp_file_name, 'w') as t_file:
            for line in code:
                t_file.write(line)

        if in_sys_path:
            # append the environment paths of the client to the current environment path.
            sys.path.extend(client_sys_path)

            # the path of the remote class is in the sys.path, we can import it directly.
            mod = __import__(new_file_name)
        else:
            module_name = module_name.lstrip('.' + os.sep).replace(os.sep, '.')
            mod = __import__(module_name, globals(), locals(), [class_name], 0)

        cls = getattr(mod, class_name)

        os.remove(tmp_file_name)
    return cls
