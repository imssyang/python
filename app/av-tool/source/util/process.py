# -*- coding: utf-8 -*-
import logging
import os
import select
import shlex
import signal
import sys
import time
import multiprocessing as mp
from collections import deque
from queue import Empty
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired


def kill_process_by_signal(pid, sig=signal.SIGKILL):
    os.kill(pid, sig)


def default_signal_handler(signum, frame):
    try:
        cpid, status = os.waitpid(-1, os.WNOHANG)
        logging.info(f"childpid: {cpid} exit status: {status}")
    except:
        pass


def register_chld_handler(self, signal_handler):
    signal.signal(signal.SIGCHLD, signal_handler)


def register_exit_handler(signal_handler):
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)


class XPipe:
    TIMEOUT_CODE = 0xFF

    def __init__(self, cmd, timeout=None, shell=False, progress_cb=None, max_cb_lines=1000):
        self.cmd = cmd
        self.timeout = timeout
        self.shell = shell
        self.progress_cb = progress_cb
        self.max_cb_lines = max_cb_lines

    def run(self):
        logging.info(f"EXEC: {self.cmd}")
        exec_result = dict(cmd=self.cmd, timeout=self.timeout)
        with Popen(self.cmd if self.shell else shlex.split(self.cmd),
            shell=self.shell, stdin=None, stdout=PIPE, stderr=PIPE,
            errors="replace", encoding="utf-8", universal_newlines=True) as process:
            if self.progress_cb:
                exec_result.update(self._manage_with_progress(process))
            else:
                exec_result.update(self._manage_without_progress(process))
        return exec_result

    def _manage_without_progress(self, process):
        try:
            exec_timeout = False
            stdout, stderr = process.communicate(timeout=self.timeout)
        except TimeoutExpired:
            exec_timeout = True
            process.kill()
            stdout, stderr = process.communicate()
        except:
            process.kill()
            raise
        finally:
            retcode = self.TIMEOUT_CODE if exec_timeout else process.poll()
        return dict(
            code=retcode,
            stdout=stdout,
            stderr=stderr,
        )

    def _manage_with_progress(self, process):
        stderr_lines = deque(maxlen=self.max_cb_lines)
        stdout_lines = deque(maxlen=self.max_cb_lines)
        r_mapper = {
            process.stdout: stdout_lines,
            process.stderr: stderr_lines,
        }
        exec_timeout = False
        try:
            begin_time = time.time()
            while process.poll() is None:
                (rpipes, _, _) = select.select([process.stdout, process.stderr], [], [], 2)

                for pipe in rpipes:
                    r_line = pipe.readline().strip()
                    if r_line:
                        r_mapper[pipe].append(r_line)
                        self._notify_progress(process, r_line)
                end_time = time.time()
                exec_time = end_time - begin_time
                if self.timeout and exec_time > self.timeout:
                    process.kill()
                    exec_timeout = True
        except:
            process.kill()
            raise
        finally:
            retcode = self.TIMEOUT_CODE if exec_timeout else process.wait()
            stdout, stderr = process.communicate()
            stdout_lines.append(stdout)
            stderr_lines.append(stderr)

        return dict(
            code=retcode,
            stdout=os.linesep.join(stdout_lines),
            stderr=os.linesep.join(stderr_lines),
        )

    def _notify_progress(self, process, content):
        try:
            progress_cb(self.cmd, process, content)
        except Exception as ex:
            logging.exception(f'Fail to execute progress callback due to {ex}')


class XProcess:
    def __init__(self, target, name, args=(), spawn=False):
        if spawn:
            ctx = mp.get_context("spawn")
            self.queue = ctx.Queue()
            self.process = ctx.Process(None, target, name, (self, *args))
        else:
            self.queue = mp.Queue()
            self.process = mp.Process(None, target, name, (self, *args))
        logging.info(f"create process: {self.name}-{self.pid}")

    def start(self):
        self.process.start()
        logging.info(f"start process: {self.name}-{self.pid}")

    def join(self):
        self.process.join()
        self.queue.close()
        logging.info(f"end process: {self.name}-{self.pid}")

    @property
    def pid(self):
        return self.process.pid

    @property
    def name(self):
        return self.process.name

    def get_msg(self):
        try:
            msg = self.queue.get(False)
        except Empty:
            return ""
        return msg

    def get_expected_msg(self, value, timeout):
        try:
            msg = self.queue.get(True, timeout)
            if msg != value:
                raise ValueError(f"{value} not match {msg} from {self.name}-{self.pid}")
        except Empty as e:
            raise TimeoutError(
                f"timeout when get msg from {self.name}-{self.pid}"
            ) from e
        return msg

    def put_msg(self, value):
        self.queue.put(value)

    def discover_child(self, name, interval):
        check_count = 0
        while True:
            if interval < check_count:
                raise TimeoutError(f"not discover {name} in {self.name}-{self.pid}")
            if VXCodeUtility.has_child_process(self.pid, name):
                break
            check_count += 1
            time.sleep(1)
        return True
