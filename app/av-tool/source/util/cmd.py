# -*- coding: utf-8 -*-
import logging
from util.process import XPipe


def get_child_processes(pid):
    result = XPipe(f"pgrep -P {pid}").run()
    if result["code"]:
        return []
    child_processes = result["stdout"].split("\n")
    child_pids = list(filter(None, child_processes))
    logging.info(f"pid: {pid} has childs: {child_pids}")
    return child_pids


def get_process_name(pid):
    result = XPipe(f"ps -q {pid} -o comm=").run()
    if result["code"]:
        logging.info(f"ps failed with: {result}")
        return ""
    return result["stdout"].strip()


def has_child_process(pid, name):
    for child_pid in get_child_processes(pid):
        child_name = get_process_name(child_pid)
        if child_name == name:
            return True
    return False


def kill_process(pid):
    result = XPipe(f"kill -s SIGKILL {pid}").run()
    if result["code"]:
        logging.info(f"kill failed with: {result}")
        return False
    return True


def kill_child_process(pid):
    result = XPipe(f"pkill --signal SIGKILL -P {pid}").run()
    if result["code"]:
        logging.info(f"pkill failed with: {result}")
        return False
    return True


def kill_process_and_child(pid):
    kill_child_process(pid)
    time.sleep(3)
    kill_process(pid)
