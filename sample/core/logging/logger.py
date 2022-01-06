import inspect
import logging
from os.path import basename


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    logger = None

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)s %(message)s",
            handlers=[
                logging.StreamHandler()
            ])
        self.logger = logging.getLogger(__name__ + '.logger')

    @staticmethod
    def __get_call_info():
        stack = inspect.stack()
        fn = stack[2][1]
        ln = stack[2][2]
        func = stack[2][3]
        return basename(fn), ln, func

    def info(self, message, *args):
        message = "{}:{}:{}] {}".format(*self.__get_call_info(), message)
        self.logger.info(message, *args)
