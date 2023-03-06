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
import requests
import subprocess
import sys
import time
import threading
import tempfile
import warnings
import zmq
from multiprocessing import Process
from parl.utils import (_IS_WINDOWS, get_free_tcp_port, get_ip_address, get_port_from_range, is_port_available,
                        kill_process, to_str)
from parl.remote.remote_constants import STATUS_TAG
from parl.remote.utils import XPARL_PYTHON

# A flag to mark if parl is started from a command line
os.environ['XPARL'] = 'True'

# Solve `Click will abort further execution because Python 3 was configured
# to use ASCII as encoding for the environment` error.

if not _IS_WINDOWS:
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except:
        pass

#TODO: this line will cause error in python2/macOS
if sys.version_info.major == 3:
    warnings.simplefilter("ignore", ResourceWarning)


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


def parse_port_range(log_server_port_range):
    try:
        re.match(r'\d*[-]\d*', log_server_port_range).span()
    except:
        raise Exception("The input log_server_port_range should be `start-end` format.")
    start, end = map(int, log_server_port_range.split('-'))
    if start > end:
        raise Exception("Start port number must be smaller than the end port number.")

    return start, end


def check_log_server_started(ip_address, port):
    started = False
    for i in range(10):
        try:
            r = requests.get("http://{}:{}/get-log".format(ip_address, port))
            if r.status_code == 400:
                started = True
                break
        except:
            pass
        time.sleep(3)
        click.echo("[{}/10] Checking status of log_server...".format(i))

    if started:
        click.echo("# Start the log server sucessfully.")
    else:
        click.echo("# Fail to start the log server.")


@click.group()
def cli():
    pass


@click.command("start", short_help="Start a master node.")
@click.option("--port", help="The port to bind to.", type=str, required=True)
@click.option("--debug", help="Start parl in the debugging mode to print all running log.", is_flag=True)
@click.option(
    "--cpu_num", type=int, help="Set number of cpu manually. If not set, it will use all "
    "cpus of this machine.")
@click.option("--gpu_cluster", help="Start as a cluster with GPUs.", is_flag=True)
@click.option("--gpu", help="Comma separated list of GPU(s) to use.", default="", type=str)
@click.option("--monitor_port", help="The port to start a cluster monitor.", type=str)
@click.option(
    "--log_server_port_range",
    help='''
    Port range (start-end) of the log server on the worker. Default: 8000-9000. 
    The worker will pick a random avaliable port in [start, end] for the log server.
    ''',
    default="8000-9000",
    type=str)
def start_master(port, gpu_cluster, cpu_num, gpu, monitor_port, debug, log_server_port_range):
    if cpu_num is not None:
        assert cpu_num >= 0, "cpu_num should be greater than or equal to 0."

    if not gpu_cluster:
        gpu = ""
    else:
        cpu_num = 0

    if debug:
        os.environ['DEBUG'] = 'True'

    if not is_port_available(port):
        raise Exception("The master address localhost:{} is already in use.".format(port))

    if monitor_port and not is_port_available(monitor_port):
        raise Exception("The input monitor port localhost:{} is already in use.".format(monitor_port))

    cpu_num = int(cpu_num) if cpu_num is not None else multiprocessing.cpu_count()
    start_file = __file__.replace('scripts.pyc', 'start.py')
    start_file = start_file.replace('scripts.py', 'start.py')
    monitor_file = __file__.replace('scripts.pyc', 'monitor.py')
    monitor_file = monitor_file.replace('scripts.py', 'monitor.py')

    if monitor_port is None:
        monitor_port = str(int(port) + 100)
        if not is_port_available(monitor_port):
            monitor_port = get_free_tcp_port()

    start, end = parse_port_range(log_server_port_range)
    log_server_port = get_port_from_range(start, end)
    while log_server_port == monitor_port or log_server_port == port:
        log_server_port = get_port_from_range(start, end)

    master_command = XPARL_PYTHON + [
        start_file,
        "--name",
        "master",
        "--port",
        port,
        "--monitor_port",
        monitor_port,
    ]
    if gpu_cluster:
        master_command.append("--gpu_cluster")
    worker_command = XPARL_PYTHON + [
        start_file, "--name", "worker", "--address", "localhost:" + str(port), "--cpu_num",
        str(cpu_num), '--log_server_port',
        str(log_server_port), "--gpu", gpu
    ]
    monitor_command = XPARL_PYTHON + [
        monitor_file, "--monitor_port",
        str(monitor_port), "--address", "localhost:" + str(port)
    ]
    if gpu_cluster:
        monitor_command.append("--gpu_cluster")

    FNULL = open(os.devnull, 'w')

    # Redirect the output to DEVNULL to solve the warning log.
    _ = subprocess.Popen(master_command, stdout=FNULL, close_fds=True)

    if cpu_num > 0 or gpu:
        # Sleep 1s for master ready
        time.sleep(1)
        _ = subprocess.Popen(worker_command, stdout=FNULL, close_fds=True)

    if _IS_WINDOWS:
        # TODO(@zenghsh3) redirecting stdout of monitor subprocess to FNULL will cause occasional failure
        tmp_file = tempfile.TemporaryFile()
        _ = subprocess.Popen(monitor_command, stdout=tmp_file, close_fds=True)
        tmp_file.close()
    else:
        _ = subprocess.Popen(monitor_command, stdout=FNULL, close_fds=True)
    FNULL.close()

    if cpu_num > 0 or gpu:
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
    monitor_is_started = False
    if _IS_WINDOWS:
        cmd = r'''wmic process where "commandline like '%remote\\monitor.py --monitor_port {} --address localhost:{}%'" get commandline /format:list | findstr /V wmic | findstr CommandLine='''.format(
            monitor_port, port)
    else:
        cmd = r'ps -ef | grep -v grep | grep remote/monitor.py\ --monitor_port\ {}\ --address\ localhost:{}'.format(
            monitor_port, port)
    for i in range(3):
        check_monitor_is_started = os.popen(cmd).read()
        if len(check_monitor_is_started) > 0:
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

        ## If you want to add more GPU resources, please call:

            xparl connect --address {}:{} --gpu 0,1,2,...

        ## If you want to shutdown the cluster, please call:

            xparl stop        
        """.format(start_info, master_ip, port, master_ip, port)
    click.echo(monitor_info)

    if cpu_num > 0 or gpu:
        check_log_server_started(master_ip, log_server_port)


@click.command("connect", short_help="Start a worker node.")
@click.option("--address", help="IP address of the master node.", required=True)
@click.option(
    "--cpu_num", type=int, help="Set number of cpu manually. If not set, it will use all "
    "cpus of this machine.")
@click.option("--gpu", help="Comma separated list of GPU(s) to use.", default="", type=str)
@click.option(
    "--log_server_port_range",
    help='''
    Port range (start-end) of the log server on the worker. Default: 8000-9000. 
    The worker will pick a random avaliable port in [start, end] for the log server.
    ''',
    default="8000-9000",
    type=str)
def start_worker(address, cpu_num, gpu, log_server_port_range):
    if cpu_num is not None:
        assert cpu_num >= 0, "cpu_num should be greater or equal to 0."

    start, end = parse_port_range(log_server_port_range)
    log_server_port = get_port_from_range(start, end)

    if not is_master_started(address):
        raise Exception("Worker can not connect to the master node, " +
                        "please check if the input address {} ".format(address) + "is correct.")
    cpu_num = str(cpu_num) if cpu_num else ''
    start_file = __file__.replace('scripts.pyc', 'start.py')
    start_file = start_file.replace('scripts.py', 'start.py')

    command = XPARL_PYTHON + [
        start_file, "--name", "worker", "--address", address, "--cpu_num",
        str(cpu_num), "--log_server_port",
        str(log_server_port), "--gpu", gpu
    ]
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(command, stdout=FNULL, close_fds=True)
    FNULL.close()

    check_log_server_started(get_ip_address(), log_server_port)


@click.command("stop", help="Exit the cluster.")
def stop():
    kill_process('remote/start.py')
    kill_process('remote/job.py')
    kill_process('remote/monitor.py')
    kill_process('remote/log_server.py')


@click.command("status")
def status():
    if _IS_WINDOWS:
        cmd = r'''wmic process where "commandline like '%remote\\start.py --name worker --address%'" get commandline /format:list | findstr /V wmic | findstr CommandLine='''
    else:
        cmd = r'ps -ef | grep remote/start.py\ --name\ worker\ --address'

    content = os.popen(cmd).read().strip()
    cpu_pattern = re.compile('--address (.*?) --cpu')
    cpu_clusters = set(cpu_pattern.findall(content))
    gpu_pattern = re.compile('--address (.*?) --cpu.*? --gpu [0-9,]+')
    gpu_clusters = set(gpu_pattern.findall(content))
    clusters = cpu_clusters | gpu_clusters
    if len(clusters) == 0:
        click.echo('No active cluster is found.')
    else:
        ctx = zmq.Context()
        status = []
        for cluster in clusters:
            if _IS_WINDOWS:
                cmd = r'''wmic process where "commandline like '%address {}%'" get commandline /format:list | findstr /V wmic | findstr CommandLine='''.format(
                    cluster)
            else:
                cmd = r'ps -ef | grep address\ {}'.format(cluster)
            content = os.popen(cmd).read()
            pattern = re.compile('--monitor_port (.*?)\n', re.S)
            monitors = pattern.findall(content)

            if len(monitors):
                monitor_port = monitors[0].split(' ')[0]
                master_address = monitors[0].split(' ')[2]
                monitor_address = "{}:{}".format(get_ip_address(), monitor_port)
                socket = ctx.socket(zmq.REQ)
                socket.setsockopt(zmq.RCVTIMEO, 10000)
                socket.connect('tcp://{}'.format(master_address))
                try:
                    socket.send_multipart([STATUS_TAG])
                    monitor_info = to_str(socket.recv_multipart()[1])
                except zmq.error.Again as e:
                    click.echo('Can not connect to cluster {}, please try later.'.format(master_address))
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
