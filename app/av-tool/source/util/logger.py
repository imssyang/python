# -*- coding: utf-8 -*-
import logging
import logging.handlers
import sys
from util.basic import MB, Singleton
from conf import setting


class _Logging(metaclass=Singleton):
    def __init__(self, name, level, filename, maxBytes, backupCount):
        self.name = name
        self.level = level
        self.format = "[%(levelname)1.1s %(asctime)s.%(msecs)03d %(threadName)s %(message)s"
        self.datefmt = "%y%m%d %H:%M:%S"
        logging.basicConfig(level=level, format=self.format, datefmt=self.datefmt, handlers=[])
        self.addStreamHandler(sys.stdout)
        self.addFileHandler(filename=filename, maxBytes=maxBytes, backupCount=backupCount)

    @property
    def logger(self):
        return logging.getLogger(self.name)

    def _addHandler(self, handler):
        handler.setLevel(self.level)
        handler.setFormatter(logging.Formatter(self.format, self.datefmt))
        self.logger.addHandler(handler)

    def addStreamHandler(self, fd):
        handler = logging.StreamHandler(fd)
        self._addHandler(handler)

    def delStreamHandler(self, fd):
        for handler in self.logger.handlers:
            if fd is handler.stream:
                self.logger.removeHandler(handler)
                break

    def addFileHandler(self, filename, maxBytes, backupCount):
        handler = logging.handlers.RotatingFileHandler(filename=filename, maxBytes=maxBytes, backupCount=backupCount)
        self._addHandler(handler)

    def showHandlers(self):
        print(self.logger.handlers)


_log = _Logging(name=None, level=logging.INFO, filename=setting.dir_log, maxBytes=MB(100), backupCount=5)
logger = _log.logger
del _Logging, _log
