# pylint: disable=consider-using-f-string

"""
python thingy.py -a 123 -b abc -c '{"a": 1}'
"""
import json
import sys
import argparse
from core.logging.logger import logger


def parse_args_f():
    logger.info(f"sys.argv: {sys.argv}")
    parser = argparse.ArgumentParser(description="description here")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1")
    parser.add_argument("-a", "--arg1", required=True, help="this is for arg1")
    parser.add_argument("-b", "--arg2", required=True, help="this is for arg2")
    parser.add_argument("-c", "--arg3", required=True, type=json.loads, help="this is for arg2")
    return parser.parse_args()


args = parse_args_f()
logger.info(f"parser: {args} {type(args.arg3)}:{args.arg3} {args.arg3['a']}")


class Thingy:
    @staticmethod
    def parse_args(args):
        logger.info(f"sys.argv: {sys.argv}\nargs: {args}")
        parser = argparse.ArgumentParser(description="description here")
        parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1")
        parser.add_argument("-a", "--arg1", required=True, help="this is for arg1")
        parser.add_argument("-b", "--arg2", required=True, help="this is for arg2")
        parser.add_argument("-c", "--arg3", required=True, help="this is for arg2")
        return parser.parse_args(args)


if __name__ == "__main__":
    parser = Thingy.parse_args(sys.argv[1:])
    logger.info("parser: {}".format(parser))
    if parser.arg1:
        logger.info(" - arg1: {}".format(parser.arg1))
    if parser.arg2:
        logger.info(" - arg2: {}".format(parser.arg2))
