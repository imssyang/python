# -*- coding: utf-8 -*-
import logging
import os
import signal
import sys
import time
import multiprocessing as mp
from queue import Empty
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired


def add_logging_handler(fd, level=logging.INFO):
    fmt = "[%(levelname)1.1s %(asctime)s %(module)-16.16s:%(lineno)4d] %(message)s"
    date_fmt = "%y%m%d %H:%M:%S"
    handler = logging.StreamHandler(fd)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, date_fmt))
    logger = logging.getLogger()
    logger.addHandler(handler)
    for handler in logger.handlers:
        # use stdout, so ignore stderr
        if "stderr" in repr(handler):
            logger.removeHandler(handler)


add_logging_handler(sys.stdout)


def current_process_info():
    cprocess = mp.current_process()
    return cprocess.name + "-" + str(cprocess.pid)


class VXCodeUtility:
    RET_TIMEOUT = "Timeout"

    @staticmethod
    def get_child_processes(pid):
        result = VXCodeUtility._run_subprocess(f"pgrep -P {pid}")
        if result["retcode"]:
            return []
        child_processes = result["stdout"].split("\n")
        return list(filter(None, child_processes))

    @staticmethod
    def get_process_name(pid):
        result = VXCodeUtility._run_subprocess(f"ps -q {pid} -o comm=")
        if result["retcode"]:
            return ""
        return result["stdout"].strip()

    @staticmethod
    def has_child_process(pid, name):
        for child_pid in VXCodeUtility.get_child_processes(pid):
            child_name = VXCodeUtility.get_process_name(child_pid)
            if child_name == name:
                return True
        return False

    @staticmethod
    def exit_signal(pid, sig=signal.SIGKILL):
        os.kill(pid, sig)

    @staticmethod
    def kill_process(pid):
        result = VXCodeUtility._run_subprocess(f"kill -s SIGKILL {pid}")
        if result["retcode"]:
            logging.info(f"kill failed with: {result}")
            return False
        return True

    @staticmethod
    def kill_child_process(pid):
        result = VXCodeUtility._run_subprocess(f"pkill --signal SIGKILL -P {pid}")
        if result["retcode"]:
            logging.info(f"pkill failed with: {result}")
            return False
        return True

    @staticmethod
    def kill_process_and_child(pid):
        VXCodeUtility.kill_child_process(pid)
        time.sleep(3)
        VXCodeUtility.kill_process(pid)

    @staticmethod
    def delete_file(path):
        result = VXCodeUtility._run_subprocess(f"rm -rf {path}")
        if result["retcode"]:
            logging.info(f"rm result: {result}")
            return False
        return True

    @staticmethod
    def exec_subprocess(cmd, timeout=None):
        result = VXCodeUtility._run_subprocess(cmd, timeout)
        if result["retcode"]:
            logging.info(f"cmd failed with: {result}")
            return False
        return True

    @classmethod
    def _run_subprocess(cls, cmd, timeout=10):
        result = {"cmd": cmd}
        with Popen(
            cmd,
            shell=True,
            stdin=None,
            stdout=PIPE,
            stderr=PIPE,
            errors="replace",
            encoding="utf-8",
            universal_newlines=True,
        ) as process:
            exec_timeout = False
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exec_timeout = True
            except:  # Including KeyboardInterrupt, communicate handled that.
                process.kill()
                raise
            finally:
                retcode = cls.RET_TIMEOUT if exec_timeout else process.poll()
        result.update(
            {
                "retcode": retcode,
                "timeout": timeout,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        return result


class VXCodeProcess:
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
