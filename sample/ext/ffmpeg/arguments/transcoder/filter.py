from __future__ import annotations
from ..common import Node, Stream
from .utils import stream_operator


class FilterableStream(Stream):
    def __init__(self, upstream_node, upstream_label, upstream_selector=None):
        super(FilterableStream, self).__init__(
            upstream_node, upstream_label, upstream_selector
        )


class FilterNode(Node):
    def __init__(self, stream_spec, name, args=[], kwargs={}):
        super(FilterNode, self).__init__(
            stream_spec=stream_spec,
            name=name,
            incoming_stream_types={FilterableStream},
            outgoing_stream_type=FilterableStream,
            args=args,
            kwargs=kwargs,
        )

    @classmethod
    def get_complex_args(cls, filter_nodes, input_nodes):
        cls._allocate_stream_names(filter_nodes, input_nodes)
        filter_specs = [node.get_spec(node, input_nodes) for node in filter_nodes]
        return ";".join(filter_specs)

    @classmethod
    def _allocate_stream_names(cls, filter_nodes, input_nodes):
        stream_name_map = {(node, None): str(i) for i, node in enumerate(input_nodes)}
        stream_count = 0
        for upstream_node in filter_nodes:
            for upstream_label, downstreams in sorted(
                upstream_node.outgoing_edge_map.items()
            ):
                if len(downstreams) > 1:
                    # TODO: automatically insert `splits` ahead of time via graph transformation.
                    raise ValueError(
                        "Encountered {} with multiple outgoing edges with same upstream "
                        "label {!r}; a `split` filter is probably required".format(
                            upstream_node, upstream_label
                        )
                    )
                stream_name_map[upstream_node, upstream_label] = f"s{stream_count}"
                stream_count += 1

    def get_spec(self, input_nodes):
        inputs = [
            _format_input_stream_name(input_nodes, edge) for edge in self.incoming_edges
        ]
        outputs = [
            _format_output_stream_name(input_nodes, edge)
            for edge in self.outgoing_edges
        ]
        filter_spec = "{}{}{}".format(
            "".join(inputs), self._get_filter(), "".join(outputs)
        )
        return filter_spec

    def _get_filter(self):
        args = self.args
        kwargs = self.kwargs
        if self.name in ("split", "asplit"):
            args = [len(self.outgoing_edges)]

        out_args = [escape_chars(x, "\\'=:") for x in args]
        out_kwargs = {}
        for k, v in list(kwargs.items()):
            k = escape_chars(k, "\\'=:")
            v = escape_chars(v, "\\'=:")
            out_kwargs[k] = v

        arg_params = [escape_chars(v, "\\'=:") for v in out_args]
        kwarg_params = ["{}={}".format(k, out_kwargs[k]) for k in sorted(out_kwargs)]
        params = arg_params + kwarg_params

        params_text = escape_chars(self.name, "\\'=:")

        if params:
            params_text += "={}".format(":".join(params))
        return escape_chars(params_text, "\\'[],;")

    @classmethod
    def escape_chars(cls, text, chars):
        """Helper function to escape uncomfortable characters."""
        text = str(text)
        chars = list(set(chars))
        if "\\" in chars:
            chars.remove("\\")
            chars.insert(0, "\\")
        for ch in chars:
            text = text.replace(ch, "\\" + ch)
        return text

    @classmethod
    def _format_input_stream_name(cls, input_nodes, edge, is_final_arg=False):
        from .input import InputNode

        stream_name_map = {(node, None): str(i) for i, node in enumerate(input_nodes)}
        prefix = stream_name_map[edge.upstream_node, edge.upstream_label]
        if not edge.upstream_selector:
            suffix = ""
        else:
            suffix = ":{}".format(edge.upstream_selector)
        if is_final_arg and isinstance(edge.upstream_node, InputNode):
            ## Special case: `-map` args should not have brackets for input nodes.
            fmt = "{}{}"
        else:
            fmt = "[{}{}]"
        return fmt.format(prefix, suffix)

    @classmethod
    def _format_output_stream_name(cls, input_nodes, edge):
        stream_name_map = {(node, None): str(i) for i, node in enumerate(input_nodes)}
        return "[{}]".format(stream_name_map[edge.upstream_node, edge.upstream_label])


def filter_operator(name=None):
    return stream_operator(stream_classes={FilterableStream}, name=name)


@filter_operator()
def output(*streams_and_filename, **kwargs):
    """Output file URL

    Syntax:
        `ffmpeg.output(stream1[, stream2, stream3...], filename, **ffmpeg_args)`

    Any supplied keyword arguments are passed to ffmpeg verbatim (e.g.
    ``t=20``, ``f='mp4'``, ``acodec='pcm'``, ``vcodec='rawvideo'``,
    etc.).  Some keyword-arguments are handled specially, as shown below.

    Args:
        video_bitrate: parameter for ``-b:v``, e.g. ``video_bitrate=1000``.
        audio_bitrate: parameter for ``-b:a``, e.g. ``audio_bitrate=200``.
        format: alias for ``-f`` parameter, e.g. ``format='mp4'``
            (equivalent to ``f='mp4'``).

    If multiple streams are provided, they are mapped to the same
    output.

    To tell ffmpeg to write to stdout, use ``pipe:`` as the filename.

    Official documentation: `Synopsis <https://ffmpeg.org/ffmpeg.html#Synopsis>`__
    """
    from .output import OutputNode

    streams_and_filename = list(streams_and_filename)
    if "filename" not in kwargs:
        if not isinstance(streams_and_filename[-1], str):
            raise ValueError("A filename must be provided")
        kwargs["filename"] = streams_and_filename.pop(-1)
    streams = streams_and_filename

    fmt = kwargs.pop("f", None)
    if fmt:
        if "format" in kwargs:
            raise ValueError("Can't specify both `format` and `f` kwargs")
        kwargs["format"] = fmt
    return OutputNode(streams, output.__name__, kwargs=kwargs).stream()


def _get_global_args(node):
    return list(node.args)
