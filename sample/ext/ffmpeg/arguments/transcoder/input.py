import copy
from ..common import Node
from .utils import convert_kwargs_to_cmd_line_args


class InputNode(Node):
    def __init__(self, name, args=[], kwargs={}):
        from .filter import FilterableStream

        super(InputNode, self).__init__(
            stream_spec=None,
            name=name,
            incoming_stream_types={},
            outgoing_stream_type=FilterableStream,
            args=args,
            kwargs=kwargs,
        )

    def get_args(self):
        if self.name == input.__name__:
            kwargs = copy.copy(self.kwargs)
            filename = kwargs.pop("filename")
            fmt = kwargs.pop("format", None)
            video_size = kwargs.pop("video_size", None)
            args = []
            if fmt:
                args += ["-f", fmt]
            if video_size:
                args += ["-video_size", "{}x{}".format(video_size[0], video_size[1])]
            args += convert_kwargs_to_cmd_line_args(kwargs)
            args += ["-i", filename]
        else:
            raise ValueError(f"Unsupported input node: {self}")
        return args


def input(filename, **kwargs):
    """Input file URL (ffmpeg ``-i`` option)

    Any supplied kwargs are passed to ffmpeg verbatim (e.g. ``t=20``,
    ``f='mp4'``, ``acodec='pcm'``, etc.).

    Official documentation: `Main options <https://ffmpeg.org/ffmpeg.html#Main-options>`
    """
    kwargs["filename"] = filename
    fmt = kwargs.pop("f", None)
    if fmt:
        if "format" in kwargs:
            raise ValueError("Can't specify both `format` and `f` kwargs")
        kwargs["format"] = fmt
    return InputNode(input.__name__, kwargs=kwargs).stream()
