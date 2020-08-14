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

import csv

__all__ = ['CSVLogger']


class CSVLogger(object):
    def __init__(self, output_file):
        """CSV Logger which can write dict result to csv file.

        Args:
            output_file(str): filename of the csv file.
        """
        self.output_file = open(output_file, "w")
        self.csv_writer = None

    def log_dict(self, result):
        """Ouput result to the csv file.

        Will create the header of the csv file automatically when the function is called for the first time.
        Ususally, the keys of the result should be the same every time you call the function.

        Args:
            result(dict)
        """
        assert isinstance(result, dict), "the input should be a dict."
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.output_file, result.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow({
            k: v
            for k, v in result.items() if k in self.csv_writer.fieldnames
        })

    def flush(self):
        self.output_file.flush()

    def close(self):
        if not self.output_file.closed:
            self.output_file.close()

    def __del__(self):
        if not self.output_file.closed:
            self.output_file.close()
