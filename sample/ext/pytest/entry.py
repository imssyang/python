# pylint: disable=no-member
import logging
import multiprocessing as mp
import os
import signal
import sys
import pytest


def pytest_main():
    pytest.main(["-c", "pytest.ini"])
    # A strange problem at pytest (6.2.5): The process that running pytest will block
    # forever, even after pytest.main has been return, so force finish it by SIGKILL.
    os.kill(mp.current_process().pid, signal.SIGKILL)


if __name__ == "__main__":
    process = mp.Process(None, pytest_main, "pytest-process")
    process.start()
    process.join()
