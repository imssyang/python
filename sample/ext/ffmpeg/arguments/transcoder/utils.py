from collections import Iterable
from flows.arguments.common.stream import Stream


def convert_kwargs_to_cmd_line_args(kwargs):
    """Helper function to build command line arguments out of dict."""
    args = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, Iterable) and not isinstance(v, str):
            for value in v:
                args.append(f"-{k}")
                if value is not None:
                    args.append(f"{value}")
            continue
        args.append(f"-{k}")
        if v is not None:
            args.append(f"{v}")
    return args


def stream_operator(stream_classes={Stream}, name=None):
    def decorator(func):
        func_name = name or func.__name__
        [setattr(stream_class, func_name, func) for stream_class in stream_classes]
        return func

    return decorator
