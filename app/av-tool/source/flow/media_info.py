import json
import logging
import util
from conf import setting
from flow.data.ffprobe import *
from plot.media_info import MediaInfoPlot
from conf import setting


class MediaInfoFlow:
    def __init__(self, filename, codec_type: str = None, stream_index: int = None, read_intervals: str = None):
        self.filename = filename
        self.codec_type = codec_type
        self.stream_index = stream_index
        self.read_intervals = read_intervals
        self.data = FFprobeData()
        self.data.format = FFprobeFormat(**self._get_format())
        if self.data.format.nb_streams:
            v_index = a_index = 0
            streams = self._get_streams(codec_type, stream_index)
            for stream in streams:
                is_video = bool("video" in stream["codec_type"])
                is_audio = bool("audio" in stream["codec_type"])
                if is_video:
                    v_stream = FFprobeVideoStream(**stream)
                    frames = self._get_frames(v_stream.codec_type, v_index)
                    for frame in frames:
                        v_frame = FFprobeVideoFrame(**frame)
                        v_stream.frames.append(v_frame)
                    packets = self._get_packets(v_stream.codec_type, v_index)
                    for packet in packets:
                        v_packet = FFprobeVideoPacket(**packet)
                        v_stream.packets.append(v_packet)
                    self.data.video.streams.append(v_stream)
                    v_index += 1
                if is_audio:
                    a_stream = FFprobeAudioStream(**stream)
                    frames = self._get_frames(a_stream.codec_type, a_index)
                    for frame in frames:
                        a_frame = FFprobeAudioFrame(**frame)
                        a_stream.frames.append(a_frame)
                    packets = self._get_packets(a_stream.codec_type, a_index)
                    for packet in packets:
                        a_packet = FFprobeAudioPacket(**packet)
                        a_stream.packets.append(a_packet)
                    self.data.audio.streams.append(a_stream)
                    a_index += 1

    def _get_select_stream_option(self, codec_type, stream_index):
        if not codec_type:
            return str()
        codec_flag = None
        if "video" in codec_type:
            codec_flag = "v"
        elif "audio" in codec_type:
            codec_flag = "a"
        if not codec_flag or stream_index is None:
            return str()
        return f"-select_streams {codec_flag}:{stream_index}"

    def _get_read_intervals_option(self):
        return f"-read_intervals {self.read_intervals}"

    def _get_option(self, codec_type, stream_index, extra_options):
        select_stream_opt = self._get_select_stream_option(codec_type, stream_index)
        read_intervals_opt = self._get_read_intervals_option()
        return f"{select_stream_opt} {read_intervals_opt} {extra_options} -of json {self.filename}"

    def _get_format(self):
        r = util.XPipe(f"{setting.bin_ffprobe} -show_format -of json {self.filename}").run()
        if r["code"]:
            logging.info(r)
            return dict()
        rd = json.loads(s=r["stdout"])
        return rd["format"]

    def _get_streams(self, codec_type, stream_index):
        options = self._get_option(codec_type, stream_index, "-show_streams")
        r = util.XPipe(f"{setting.bin_ffprobe} {options}").run()
        if r["code"]:
            logging.info(r)
            return list()
        rd = json.loads(s=r["stdout"])
        return rd["streams"]

    def _get_frames(self, codec_type, stream_index):
        options = self._get_option(codec_type, stream_index, "-show_frames")
        r = util.XPipe(f"{setting.bin_ffprobe} {options}").run()
        if r["code"]:
            logging.info(r)
            return dict()
        rd = json.loads(s=r["stdout"])
        return rd["frames"]

    def _get_packets(self, codec_type, stream_index):
        options = self._get_option(codec_type, stream_index, "-show_packets")
        r = util.XPipe(f"{setting.bin_ffprobe} {options}").run()
        if r["code"]:
            logging.info(r)
            return dict()
        rd = json.loads(s=r["stdout"])
        return rd["packets"]


if __name__ == "__main__":
    flow = MediaInfoFlow("/opt/ffmpeg/sample/dota2/10-20.flv", read_intervals="%+#30")
    #logging.info(flow.data.dict())
    plot = MediaInfoPlot(flow.data)
    plot.show(10, 10)
    plot.save(10, 10, f"{setting.dir_workspace}/plot.png")

