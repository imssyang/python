import time
from multiprocessing import Process
import os


def info(title):
    print(title)
    print("module name:", __name__)
    print("parent process:", os.getppid())
    print("process id:", os.getpid())


def f(name):
    time.sleep(10)
    info("function f")
    print("hello", name)


if __name__ == "__main__":
    info("main line")
    p = Process(target=f, args=("bob",))
    p.start()
    time.sleep(3)
    print(f"{p}")
    # p.kill()
    # p.join()
