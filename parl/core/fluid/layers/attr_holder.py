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

import six
from copy import deepcopy

__all__ = ['AttrHolder']


class AttrHolder(object):
    """ Mainly used for maintaining all ParamAttr in a parl.layers.LayerFunc
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: {name:attr}
        """
        self._attrs_dict = {}
        for k, v in six.iteritems(kwargs):
            self._add_attr(k, v)

    def _add_attr(self, name, attr):
        assert name not in self._attrs_dict
        self._attrs_dict[name] = attr

    def __setattr__(self, name, attr):
        if not name.startswith('_'):
            self._add_attr(name, attr)
        else:
            # self._attrs_dict
            super(AttrHolder, self).__setattr__(name, attr)

    def __getattr__(self, name):
        if name in self._attrs_dict.keys():
            return self._attrs_dict[name]
        else:
            return None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in six.iteritems(self.__dict__):
            setattr(result, k, deepcopy(v, memo))
        return result

    def sorted(self):
        """
        Returns:
            list of all attrs, which is sorted by key
        """
        return [self._attrs_dict[k] for k in sorted(self._attrs_dict.keys())]

    def tolist(self):
        """
        Returns:
            list of all attrs
        """
        return list(six.itervalues(self._attrs_dict))
