# -*- coding: utf-8 -*-
from collections import namedtuple
from .utils import get_hash, get_hash_int
from .stream import Stream, IncomingStreams


Edge = namedtuple(
    "Edge",
    [
        "downstream_node",
        "downstream_label",
        "upstream_node",
        "upstream_label",
        "upstream_selector",
    ],
)


class Node(object):
    """Node in a directed-acyclic graph (DAG).

    Nodes and Edges:
        DagNodes are connected by edges. An edge connects two nodes with a label for
        each side:
         - ``upstream_node``: upstream/parent node
         - ``upstream_label``: label on the outgoing side of the upstream node
         - ``downstream_node``: downstream/child node
         - ``downstream_label``: label on the incoming side of the downstream node

        For example, DagNode A may be connected to DagNode B with an edge labelled
        "foo" on A's side, and "bar" on B's side:

           _____               _____
          |     |             |     |
          |  A  >[foo]---[bar]>  B  |
          |_____|             |_____|

        Edge labels may be integers or strings, and nodes cannot have more than one
        incoming edge with the same label.

        DagNodes may have any number of incoming edges and any number of outgoing
        edges.  DagNodes keep track only of their incoming edges, but the entire graph
        structure can be inferred by looking at the furthest downstream nodes and
        working backwards.
    """

    def __init__(
        self,
        stream_spec,
        name,
        incoming_stream_types,
        outgoing_stream_type,
        args=[],
        kwargs={},
    ):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.__sspec = IncomingStreams(stream_spec)
        self.__outgoing_stream_type = outgoing_stream_type
        self.__incoming_stream_types = incoming_stream_types
        self.__incoming_edge_map = self.__get_incoming_edge_map()
        self.__backtrack_nodes = None
        self.__backtrack_outgoing_edge_maps = None
        self.__hash = self.__get_hash()

    @property
    def __upstream_hashes(self):
        hashes = []
        for downstream_label, upstream_info in list(self.incoming_edge_map.items()):
            upstream_node, upstream_label, upstream_selector = upstream_info
            hashes += [
                hash(x)
                for x in [
                    downstream_label,
                    upstream_node,
                    upstream_label,
                    upstream_selector,
                ]
            ]
        return hashes

    @property
    def __inner_hash(self):
        props = {"args": self.args, "kwargs": self.kwargs}
        return get_hash(props)

    def __get_hash(self):
        return get_hash_int(self.__upstream_hashes + [self.__inner_hash])

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def short_hash(self):
        return "{:x}".format(abs(hash(self)))[:12]

    def long_repr(self, include_hash=True):
        formatted_props = ["{!r}".format(arg) for arg in self.args]
        formatted_props += [
            "{}={!r}".format(key, self.kwargs[key]) for key in sorted(self.kwargs)
        ]
        out = "{}({})".format(self.name, ", ".join(formatted_props))
        if include_hash:
            out += " <{}>".format(self.short_hash)
        return out

    def __repr__(self):
        """Return a full string representation of the node."""
        return self.long_repr()

    def __getitem__(self, item):
        """Create an outgoing stream originating from this node; syntactic sugar for
        ``self.stream(label)``. It can also be used to apply a selector: e.g.
        ``node[0:'a']`` returns a stream with label 0 and selector ``'a'``, which is
        the same as ``node.stream(label=0, selector='a')``.

        Example:
            Process the audio and video portions of a stream independently::

                input = xxx.input('in.mp4')
                audio = input[:'a'].filter("aecho", 0.8, 0.9, 1000, 0.3)
                video = input[:'v'].hflip()
                out = xxx.output(audio, video, 'out.mp4')
        """
        if isinstance(item, slice):
            return self.stream(label=item.start, selector=item.stop)
        else:
            return self.stream(label=item)

    def stream(self, label=None, selector=None):
        return self.__outgoing_stream_type(self, label, upstream_selector=selector)

    def __get_incoming_edge_map(self):
        incoming_edge_map = {}
        stream_map = self.__sspec.stream_map
        for downstream_label, upstream in list(stream_map.items()):
            incoming_edge_map[downstream_label] = (
                upstream.node,
                upstream.label,
                upstream.selector,
            )
        return incoming_edge_map

    @property
    def incoming_edges(self):
        edges = []
        downstream_node = self
        for downstream_label, upstream_info in list(self.incoming_edge_map.items()):
            upstream_node, upstream_label, upstream_selector = upstream_info
            edges += [
                Edge(
                    downstream_node,
                    downstream_label,
                    upstream_node,
                    upstream_label,
                    upstream_selector,
                )
            ]
        return edges

    @property
    def incoming_edge_map(self):
        """Provides information about all incoming edges that connect to this node."""
        return self.__incoming_edge_map

    def __backtrack(self):
        marked_nodes = []
        sorted_nodes = []
        outgoing_edge_maps = {}

        def visit(
            upstream_node,
            upstream_label,
            downstream_node,
            downstream_label,
            downstream_selector=None,
        ):
            if upstream_node in marked_nodes:
                raise RuntimeError("Graph is not a DAG")

            if downstream_node is not None:
                outgoing_edge_map = outgoing_edge_maps.get(upstream_node, {})
                outgoing_edge_infos = outgoing_edge_map.get(upstream_label, [])
                outgoing_edge_infos += [
                    (downstream_node, downstream_label, downstream_selector)
                ]
                outgoing_edge_map[upstream_label] = outgoing_edge_infos
                outgoing_edge_maps[upstream_node] = outgoing_edge_map

            if upstream_node not in sorted_nodes:
                marked_nodes.append(upstream_node)
                for edge in upstream_node.incoming_edges:
                    visit(
                        edge.upstream_node,
                        edge.upstream_label,
                        edge.downstream_node,
                        edge.downstream_label,
                        edge.upstream_selector,
                    )
                marked_nodes.remove(upstream_node)
                sorted_nodes.append(upstream_node)

        unmarked_nodes = [(node, None) for node in self.__sspec.nodes]
        while unmarked_nodes:
            upstream_node, upstream_label = unmarked_nodes.pop()
            visit(upstream_node, upstream_label, None, None)
        for upstream_node in sorted_nodes:
            upstream_node.__backtrack_nodes = sorted_nodes
            upstream_node.__backtrack_outgoing_edge_maps = outgoing_edge_maps
        return sorted_nodes, outgoing_edge_maps

    @property
    def outgoing_edges(self):
        edges = []
        upstream_node = self
        for upstream_label, downstream_infos in sorted(self.outgoing_edge_map.items()):
            for downstream_info in downstream_infos:
                downstream_node, downstream_label, downstream_selector = downstream_info
                edges += [
                    DagEdge(
                        downstream_node,
                        downstream_label,
                        upstream_node,
                        upstream_label,
                        downstream_selector,
                    )
                ]
        return edges

    @property
    def outgoing_edge_map(self):
        if not self.__backtrack_outgoing_edge_maps:
            _, self.__backtrack_outgoing_edge_maps = self.__backtrack()
        return self.__backtrack_outgoing_edge_maps[self]

    @property
    def backtrack_nodes(self):
        if not self.__backtrack_nodes:
            self.__backtrack_nodes, _ = self.__backtrack()
        return self.__backtrack_nodes
