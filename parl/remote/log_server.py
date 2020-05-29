#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import linecache
import os

from flask import Flask, current_app, jsonify, make_response, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route(
    "/get-log", methods=[
        'GET',
    ])
def get_log():
    '''
    args: 
        job_id: id of the remote job
    response:
        log: newest `LINE_NUM` lines of the log file
    '''
    try:
        job_id = request.args['job_id']
    except:
        return make_response(
            jsonify(message="No job_id provided, please check your request."),
            400)

    log_dir = current_app.config.get('LOG_DIR')
    log_dir = os.path.expanduser(log_dir)
    log_file_path = os.path.join(log_dir, job_id, 'stdout.log')
    if not os.path.isfile(log_file_path):
        return make_response(
            jsonify(message="Log not exsits, please check your job_id"), 400)
    else:
        line_num = current_app.config.get('LINE_NUM')
        linecache.checkcache(log_file_path)
        log_content = ''.join(linecache.getlines(log_file_path)[-line_num:])
        return make_response(
            jsonify(message="Log exsits, content in log", log=log_content),
            200)


@app.route(
    '/download-log', methods=[
        'GET',
    ])
def download_log():
    '''
    args:
        job_id: the id of the remote job
    response:
        log: log file
    '''
    try:
        job_id = request.args['job_id']
    except:
        return make_response(
            jsonify(message="No job_id provided, please check your request."),
            400)
    log_dir = current_app.config.get('LOG_DIR')
    log_dir = os.path.expanduser(log_dir)
    log_file_path = os.path.join(log_dir, job_id, 'stdout.log')
    if not os.path.isfile(log_file_path):
        return make_response(
            jsonify(message="Log not exsits, please check your job_id"), 400)
    else:
        return send_file(log_file_path, as_attachment=True)


if __name__ == "__main__":
    import logging
    log = logging.getLogger('werkzeug')
    log.disabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--log_dir', required=True, type=str)
    parser.add_argument('--line_num', required=True, type=int)
    args = parser.parse_args()

    app.config.from_mapping(
        LOG_DIR=args.log_dir,
        LINE_NUM=args.line_num,
    )

    app.run(host="0.0.0.0", port=args.port)
