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

import click
import locale
import os
import subprocess
import threading
import warnings
from multiprocessing import Process

# A flag to mark if parl is started from a command line
os.environ['XPARL'] = 'True'

# Solve `Click will abort further execution because Python 3 was configured
# to use ASCII as encoding for the environment` error.
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

warnings.simplefilter("ignore", ResourceWarning)


def is_port_in_use(port):
    """ Check if a port is used.

    True if the port is not available. Otherwise, this port can be used for
    connection.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', int(port))) == 0


def is_master_started(address):
    import zmq
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.linger = 0
    socket.setsockopt(zmq.RCVTIMEO, 500)
    socket.connect("tcp://{}".format(address))
    socket.send_multipart([b'[NORMAL]'])
    try:
        _ = socket.recv_multipart()
        socket.close(0)
        return True
    except zmq.error.Again as e:
        socket.close(0)
        return False


@click.group()
def cli():
    pass


@click.command("start", short_help="Start a master node.")
@click.option("--port", help="The port to bind to.", type=str, required=True)
@click.option(
    "--cpu_num",
    type=int,
    help="Set number of cpu manually. If not set, it will use all "
    "cpus of this machine.")
def start_master(port, cpu_num):
    if is_port_in_use(port):
        raise Exception(
            "The master address localhost:{} already in use.".format(port))
    cpu_num = str(cpu_num) if cpu_num else ''
    command = [
        "python", "{}/start.py".format(__file__[:-11]), "--name", "master",
        "--port", port
    ]
    p = subprocess.Popen(command)

    command = [
        "python", "{}/start.py".format(__file__[:-11]), "--name", "worker",
        "--address", "localhost:" + str(port), "--cpu_num",
        str(cpu_num)
    ]
    p = subprocess.Popen(command)


@click.command("connect", short_help="Start a worker node.")
@click.option(
    "--address", help="IP address of the master node.", required=True)
@click.option(
    "--cpu_num",
    type=int,
    help="Set number of cpu manually. If not set, it will use all "
    "cpus of this machine.")
def start_worker(address, cpu_num):
    if not is_master_started(address):
        raise Exception("Worker can not connect to the master node, " +
                        "please check if the input address {} ".format(
                            address) + "is correct.")
    cpu_num = str(cpu_num) if cpu_num else ''
    command = [
        "python", "{}/start.py".format(__file__[:-11]), "--name", "worker",
        "--address", address, "--cpu_num",
        str(cpu_num)
    ]
    p = subprocess.Popen(command)


@click.command("stop", help="Exit the cluster.")
def stop():
    command = ("pkill -f remote/start.py")
    subprocess.call([command], shell=True)
    command = ("pkill -f job.py")
    p = subprocess.call([command], shell=True)


cli.add_command(start_worker)
cli.add_command(start_master)
cli.add_command(stop)


def main():
    return cli()


if __name__ == "__main__":
    main()
