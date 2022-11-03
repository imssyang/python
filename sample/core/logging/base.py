import logging
import sys


class Logging:
    def __init__(self, name=None):
        self.name = name
        self.level = logging.INFO
        self.fmt = (
            "[%(levelname)1.1s %(asctime)s.%(msecs)03d %(threadName)s %(message)s"
        )
        self.datefmt = "%y%m%d %H:%M:%S"
        logging.basicConfig(level=self.level, format=self.fmt, datefmt=self.datefmt)
        self.add_handler(logging.getLogger(name), sys.stdout)
        self.del_handler(logging.getLogger(name), sys.stderr)
        self.list_logger()

    def add_handler(self, logger, fd):
        handler = logging.StreamHandler(fd)
        handler.setLevel(self.level)
        handler.setFormatter(logging.Formatter(self.fmt, self.datefmt))
        logger.addHandler(handler)

    def del_handler(self, logger, fd):
        for handler in logger.handlers:
            if fd is handler.stream:
                logger.removeHandler(handler)
                break

    def list_logger(self):
        logger = logging.getLogger(self.name)
        print(logger.handlers)


if __name__ == "__main__":
    Logging()
    byte_string = "\xc3\xb4"
    unicode_string = "щ"
    logging.info(f"11111")
    logging.info(f"中文")
    logging.info(f"{byte_string}")
    logging.info(f"{unicode_string}")
