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
import multiprocessing
import os
import random
import re
import socket
import subprocess
import sys
import time
import threading
import warnings
import zmq
from multiprocessing import Process
from parl.utils import get_ip_address, to_str
from parl.remote.remote_constants import STATUS_TAG

# A flag to mark if parl is started from a command line
os.environ['XPARL'] = 'True'

# Solve `Click will abort further execution because Python 3 was configured
# to use ASCII as encoding for the environment` error.
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

#TODO: this line will cause error in python2/macOS
if sys.version_info.major == 3:
    warnings.simplefilter("ignore", ResourceWarning)


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return str(port)


def is_port_available(port):
    """ Check if a port is used.

    True if the port is available for connection.
    """
    port = int(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    available = sock.connect_ex(('localhost', port))
    sock.close()
    return available


def is_master_started(address):
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
    "--debug",
    help="Start parl in the debugging mode to print all running log.",
    is_flag=True)
@click.option(
    "--cpu_num",
    type=int,
    help="Set number of cpu manually. If not set, it will use all "
    "cpus of this machine.")
@click.option(
    "--monitor_port", help="The port to start a cluster monitor.", type=str)
def start_master(port, cpu_num, monitor_port, debug):
    if debug:
        os.environ['DEBUG'] = 'True'

    if not is_port_available(port):
        raise Exception(
            "The master address localhost:{} is already in use.".format(port))

    if monitor_port and not is_port_available(monitor_port):
        raise Exception(
            "The input monitor port localhost:{} is already in use.".format(
                monitor_port))

    cpu_num = int(
        cpu_num) if cpu_num is not None else multiprocessing.cpu_count()
    start_file = __file__.replace('scripts.pyc', 'start.py')
    start_file = start_file.replace('scripts.py', 'start.py')
    monitor_port = monitor_port if monitor_port else get_free_tcp_port()

    master_command = [
        sys.executable, start_file, "--name", "master", "--port", port
    ]
    worker_command = [
        sys.executable, start_file, "--name", "worker", "--address",
        "localhost:" + str(port), "--cpu_num",
        str(cpu_num)
    ]
    monitor_command = [
        sys.executable, '{}/monitor.py'.format(__file__[:__file__.rfind('/')]),
        "--monitor_port",
        str(monitor_port), "--address", "localhost:" + str(port)
    ]

    FNULL = open(os.devnull, 'w')

    # Redirect the output to DEVNULL to solve the warning log.
    _ = subprocess.Popen(
        master_command, stdout=FNULL, stderr=subprocess.STDOUT)
    if cpu_num > 0:
        _ = subprocess.Popen(
            worker_command, stdout=FNULL, stderr=subprocess.STDOUT)
    _ = subprocess.Popen(
        monitor_command, stdout=FNULL, stderr=subprocess.STDOUT)
    FNULL.close()

    if cpu_num > 0:
        monitor_info = """
            # The Parl cluster is started at localhost:{}.

            # A local worker with {} CPUs is connected to the cluster.    

            # Starting the cluster monitor...""".format(
            port,
            cpu_num,
        )
    else:
        monitor_info = """
            # The Parl cluster is started at localhost:{}.

            # Starting the cluster monitor...""".format(port)
    click.echo(monitor_info)

    # check if monitor is started
    cmd = r'ps -ef | grep remote/monitor.py\ --monitor_port\ {}\ --address\ localhost:{}'.format(
        monitor_port, port)

    monitor_is_started = False
    for i in range(3):
        check_monitor_is_started = os.popen(cmd).read().strip().split('\n')
        if len(check_monitor_is_started) == 2:
            monitor_is_started = True
            break
        time.sleep(3)
    master_ip = get_ip_address()
    if monitor_is_started:
        start_info = """
        ## If you want to check cluster status, please view:

            http://{}:{}

        or call:

            xparl status""".format(master_ip, monitor_port)
    else:
        start_info = "# Fail to start the cluster monitor."

    monitor_info = """
        {}

        ## If you want to add more CPU resources, please call:

            xparl connect --address {}:{}

        ## If you want to shutdown the cluster, please call:

            xparl stop        
        """.format(start_info, master_ip, port)
    click.echo(monitor_info)


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
        sys.executable, "{}/start.py".format(__file__[:-11]), "--name",
        "worker", "--address", address, "--cpu_num",
        str(cpu_num)
    ]
    p = subprocess.Popen(command)


@click.command("stop", help="Exit the cluster.")
def stop():
    command = (
        "ps aux | grep remote/start.py | awk '{print $2}' | xargs kill -9")
    subprocess.call([command], shell=True)
    command = (
        "ps aux | grep remote/job.py | awk '{print $2}' | xargs kill -9")
    subprocess.call([command], shell=True)
    command = (
        "ps aux | grep remote/monitor.py | awk '{print $2}' | xargs kill -9")
    subprocess.call([command], shell=True)


@click.command("status")
def status():
    cmd = r'ps -ef | grep remote/start.py\ --name\ worker\ --address'
    content = os.popen(cmd).read().strip()
    pattern = re.compile('--address (.*?) --cpu')
    clusters = set(pattern.findall(content))
    if len(clusters) == 0:
        click.echo('No active cluster is found.')
    else:
        ctx = zmq.Context()
        status = []
        for cluster in clusters:
            cmd = r'ps -ef | grep address\ {}'.format(cluster)
            content = os.popen(cmd).read()
            pattern = re.compile('--monitor_port (.*?)\n', re.S)
            monitors = pattern.findall(content)

            if len(monitors):
                monitor_port, _, master_address = monitors[0].split(' ')
                monitor_address = "{}:{}".format(get_ip_address(),
                                                 monitor_port)
                socket = ctx.socket(zmq.REQ)
                socket.setsockopt(zmq.RCVTIMEO, 10000)
                socket.connect('tcp://{}'.format(master_address))
                try:
                    socket.send_multipart([STATUS_TAG])
                    monitor_info = to_str(socket.recv_multipart()[1])
                except zmq.error.Again as e:
                    click.echo(
                        'Can not connect to cluster {}, please try later.'.
                        format(master_address))
                    socket.close(0)
                    continue
                msg = """
            # Cluster {} {}

            # If you want to check cluster status, please view: http://{}
            """.format(master_address, monitor_info, monitor_address)
                status.append(msg)
                socket.close(0)
            else:
                msg = """
            # Cluster {} fails to start the cluster monitor.
                """.format(cluster)
                status.append(msg)

        for monitor_status in status:
            click.echo(monitor_status)


cli.add_command(start_worker)
cli.add_command(start_master)
cli.add_command(stop)
cli.add_command(status)


def main():
    return cli()


if __name__ == "__main__":
    main()
