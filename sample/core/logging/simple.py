import logging
import logging.handlers


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)1.1s %(asctime)s.%(msecs)03d %(message)s",
        datefmt="%y%m%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                "setup.log", maxBytes=100000000, backupCount=5
            ),
        ],
    )


init_logging()
logging.info(123)
