# -*- coding: utf-8 -*-
from .utils import get_hash_int


class Stream(object):
    """Represents the outgoing edge of an upstream node;
    may be used to create more downstream nodes.
    """

    def __init__(
        self, upstream_node, upstream_label: str, upstream_selector: str = None
    ):
        self.node = upstream_node
        self.label = upstream_label
        self.selector = upstream_selector

    def __hash__(self):
        return get_hash_int([hash(self.node), hash(self.label)])

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        node_repr = self.node.long_repr(include_hash=False)
        selector = ""
        if self.selector:
            selector = ":{}".format(self.selector)
        out = "{}[{!r}{}] <{}>".format(
            node_repr, self.label, selector, self.node.short_hash
        )
        return out

    def __getitem__(self, index):
        """
        Select a component (audio, video) of the stream.

        ``stream.audio`` is a shorthand for ``stream['a']``.
        ``stream.video`` is a shorthand for ``stream['v']``.

        Example:
            Process the audio and video portions of a stream independently::

                input = xxx.input('in.mp4')
                audio = input['a'].filter("aecho", 0.8, 0.9, 1000, 0.3)
                video = input['v'].hflip()
                out = xxx.output(audio, video, 'out.mp4')
        """
        if self.selector is not None:
            raise ValueError(f"Stream already has a selector: {self}")
        elif not isinstance(index, str):
            raise TypeError("Expected string index (e.g. 'a'); got {!r}".format(index))
        return self.node.stream(label=self.label, selector=index)

    @property
    def audio(self):
        return self["a"]

    @property
    def video(self):
        return self["v"]


class IncomingStreams(object):
    """Trace upstream nodes according to the downstream label;
    used to analyse ``stream_spec`` param.
    """

    def __init__(self, stream_spec):
        self.__upstream_map = self.__get_down2upstream_map(stream_spec)
        self.__upstream_nodes = self.__get_upstream_nodes(self.__upstream_map)

    def __get_down2upstream_map(self, stream_spec):
        if stream_spec is None:
            stream_map = {}
        elif isinstance(stream_spec, Stream):
            stream_map = {None: stream_spec}
        elif isinstance(stream_spec, (list, tuple)):
            stream_map = dict(enumerate(stream_spec))
        elif isinstance(stream_spec, dict):
            stream_map = stream_spec
        else:
            stream_map = {}
        return stream_map

    def __get_upstream_nodes(self, stream_map):
        nodes = []
        for stream in list(stream_map.values()):
            if not isinstance(stream, Stream):
                raise TypeError("Expected Stream; got {}".format(type(stream)))
            nodes.append(stream.node)
        return nodes

    @property
    def stream_map(self):
        return self.__upstream_map

    @property
    def nodes(self):
        return self.__upstream_nodes
