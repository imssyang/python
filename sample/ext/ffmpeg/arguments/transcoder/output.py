from __future__ import annotations
import copy
import operator
from collections import Iterable
from functools import reduce
from ..common import Node, Stream, IncomingStreams
from .input import InputNode
from .filter import FilterableStream, FilterNode
from .utils import stream_operator
from .utils import convert_kwargs_to_cmd_line_args


class OutputStream(Stream):
    def __init__(self, upstream_node, upstream_label, upstream_selector=None):
        super(OutputStream, self).__init__(
            upstream_node,
            upstream_label,
            upstream_selector,
        )


class OutputNode(Node):
    def __init__(self, stream, name, args=[], kwargs={}):
        super(OutputNode, self).__init__(
            stream_spec=stream,
            name=name,
            incoming_stream_types={FilterableStream},
            outgoing_stream_type=OutputStream,
            args=args,
            kwargs=kwargs,
        )

    def get_args(self, input_nodes):
        from .filter import output

        if self.name != output.__name__:
            raise ValueError(f"Unsupported output node: {self}")

        args = []
        if len(self.incoming_edges) == 0:
            raise ValueError(f"Output node {self} has no mapped streams")

        for edge in self.incoming_edges:
            stream_name = self._format_input_stream_name(
                input_nodes, edge, is_final_arg=True
            )
            if stream_name != "0" or len(self.incoming_edges) > 1:
                args += ["-map", stream_name]

        kwargs = copy.copy(self.kwargs)
        filename = kwargs.pop("filename")
        if "format" in kwargs:
            args += ["-f", kwargs.pop("format")]
        if "video_bitrate" in kwargs:
            args += ["-b:v", str(kwargs.pop("video_bitrate"))]
        if "audio_bitrate" in kwargs:
            args += ["-b:a", str(kwargs.pop("audio_bitrate"))]
        if "video_size" in kwargs:
            video_size = kwargs.pop("video_size")
            if not isinstance(video_size, str) and isinstance(video_size, Iterable):
                video_size = "{}x{}".format(video_size[0], video_size[1])
            args += ["-video_size", video_size]
        args += convert_kwargs_to_cmd_line_args(kwargs)
        args += [filename]
        return args

    @classmethod
    def _format_input_stream_name(cls, input_nodes, edge, is_final_arg=False):
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


class MergeOutputsNode(Node):
    def __init__(self, streams, name):
        super(MergeOutputsNode, self).__init__(
            stream_spec=streams,
            name=name,
            incoming_stream_types={OutputStream},
            outgoing_stream_type=OutputStream,
        )


def output_operator(name=None):
    return stream_operator(stream_classes={OutputStream}, name=name)


@output_operator()
def get_args(stream_spec, overwrite_output=False):
    """Build command-line arguments to be passed to transcoder."""
    downstream = MergeOutputsNode(stream_spec, get_args.__name__).stream()
    backtrack_nodes = IncomingStreams(downstream).nodes[0].backtrack_nodes
    input_nodes = [node for node in backtrack_nodes if isinstance(node, InputNode)]
    # global_nodes = [node for node in backtrack_nodes if isinstance(node, GlobalNode)]
    filter_nodes = [node for node in backtrack_nodes if isinstance(node, FilterNode)]
    output_nodes = [node for node in backtrack_nodes if isinstance(node, OutputNode)]

    args = []
    args += reduce(operator.add, [node.get_args() for node in input_nodes])
    filter_arg = FilterNode.get_complex_args(filter_nodes, input_nodes)
    if filter_arg:
        args += ["-filter_complex", filter_arg]
    args += reduce(operator.add, [node.get_args(input_nodes) for node in output_nodes])
    # args += reduce(operator.add, [_get_global_args(node) for node in global_nodes], [])
    if overwrite_output:
        args += ["-y"]
    return args


@output_operator()
def compile(stream_spec, cmd="bvc_vod_transcoder"):
    if isinstance(cmd, str):
        cmd = [cmd]
    return cmd + get_args(stream_spec)
