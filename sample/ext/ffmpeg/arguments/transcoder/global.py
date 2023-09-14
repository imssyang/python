class GlobalNode(Node):
    def __init__(self, stream, name, args=[], kwargs={}):
        super(GlobalNode, self).__init__(
            stream_spec=stream,
            name=name,
            incoming_stream_types={InputStream},
            outgoing_stream_type=InputStream,
            args=args,
            kwargs=kwargs,
        )


@output_operator()
def global_args(stream, *args):
    """Add extra global command-line argument(s), e.g. ``-progress``."""
    return GlobalNode(stream, global_args.__name__, args).stream()


def global_(filename, **kwargs):
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
