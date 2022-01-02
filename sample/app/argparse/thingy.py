'''
python thingy.py -a 123 -b abc
'''
import sys
import argparse
from util.mlogger import mlog


def parse_args_f():
    mlog.info(f"sys.argv: {sys.argv}")
    parser = argparse.ArgumentParser(description='description here')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-a', '--arg1', required=True, help='this is for arg1')
    parser.add_argument('-b', '--arg2', required=True, help='this is for arg2')
    return parser.parse_args()


parser_f = parse_args_f()
mlog.info(f"parser_f: {parser_f}")


class Thingy:
    @staticmethod
    def parse_args(args):
        mlog.info(f"sys.argv: {sys.argv}\nargs: {args}")
        parser = argparse.ArgumentParser(description='description here')
        parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
        parser.add_argument('-a', '--arg1', required=True, help='this is for arg1')
        parser.add_argument('-b', '--arg2', required=True, help='this is for arg2')
        return parser.parse_args(args)


if __name__ == '__main__':
    parser = Thingy.parse_args(sys.argv[1:])
    mlog.info('parser: {}'.format(parser))
    if parser.arg1:
        mlog.info(' - arg1: {}'.format(parser.arg1))
    if parser.arg2:
        mlog.info(' - arg2: {}'.format(parser.arg2))
