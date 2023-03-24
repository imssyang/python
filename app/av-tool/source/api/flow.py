import json
import logging
from fractions import Fraction
from flask import abort
from flask import Blueprint
from flask import render_template
from flask import request
from flow.media_info import MediaInfoFlow

bp = Blueprint("flow", __name__, url_prefix="/flow")


class ChartOption:
    def __init__(self, data):
        self.data = data
        logging.info(data.json())

    def getRenderData(self):
        renderData = dict(data=self.data.json(), video={}, audio={}, avmix={})
        for index, stream in enumerate(self.data.video.streams):
            renderStream = dict()
            frameTypeOption = self._getVPictTypeOption(index)
            renderStream.update({f"frame_type_option": frameTypeOption})
            renderData["video"][stream.index] = renderStream
        for index, stream in enumerate(self.data.audio.streams):
            renderStream = dict()
            sampleFmtOption = self._getASampleFmtOption(index)
            renderStream.update({f"sample_fmt_option": sampleFmtOption})
            renderData["audio"][stream.index] = renderStream
        if len(self.data.video.streams) > 0 and len(self.data.audio.streams) > 0:
            videoIndex = audioIndex = 0
            renderStream = {f"pts_contrast_option": self._getAVPtsContrastOption(videoIndex, audioIndex)}
            renderData["avmix"][f"{videoIndex}-{audioIndex}"] = renderStream
        return renderData

    def _getPktDurationInFrame(self, stream):
        for frame in stream.frames:
            if frame.pkt_duration and frame.pkt_duration > 0:
                return frame.pkt_duration
        return 0

    def _getChannelsInFrame(self, stream):
        for frame in stream.frames:
            if frame.channels and frame.channels > 0:
                return frame.channels
        return 0

    def _getPktSizeListInFrame(self, stream, pict_type):
        return [frame.pkt_size if pict_type == frame.pict_type else '-' for frame in stream.frames]

    def _getPtsDiffListInFrame(self, stream):
        result = list()
        for index, frame in enumerate(stream.frames):
            if index == 0:
                continue
            result.append(stream.frames[index].pkt_pts - stream.frames[index-1].pkt_pts)
        return result

    def _getSizeListInPacket(self, stream):
        return [packet.size for packet in stream.packets]

    def _getVPictTypeOption(self, stream_index):
        stream = self.data.video.streams[stream_index]
        return dict(
            title=dict(
                text=f"stream-{stream.index} | {stream.codec_name} | {stream.profile} | {stream.width}x{stream.height} | "
                     f"{stream.pix_fmt} | {int(Fraction(stream.r_frame_rate))}fps | {int(stream.bit_rate/1000)}kb/s | "
                     f"{stream.time_base} | {stream.start_pts}pts | {self._getPktDurationInFrame(stream)} delta"
            ),
            xAxis=dict(
                name='frame_index',
                data=[index for index, _ in enumerate(stream.frames)]
            ),
            yAxis=dict(
                name='pkt_size'
            ),
            series=[
                dict(data=self._getPktSizeListInFrame(stream, "I")),
                dict(data=self._getPktSizeListInFrame(stream, "P")),
                dict(data=self._getPktSizeListInFrame(stream, "B")),
                dict(data=self._getSizeListInPacket(stream)),
            ]
        )

    def _getASampleFmtOption(self, stream_index):
        stream = self.data.audio.streams[stream_index]
        return dict(
            title=dict(
                text=f"stream-{stream.index} | {stream.codec_name} | {stream.profile} | {stream.sample_fmt} | {stream.channel_layout} | "
                     f"{self._getChannelsInFrame(stream)} chan | {int(stream.sample_rate)} Hz | {int(stream.bit_rate/1000)}kb/s | "
                     f"{stream.time_base} | {stream.start_pts}pts | {self._getPktDurationInFrame(stream)} delta"
            ),
            xAxis=dict(
                name='frame_index',
                data=[index for index, _ in enumerate(stream.frames)]
            ),
            yAxis=dict(
                name='pkt_size'
            ),
            series=[
                dict(data=[frame.pkt_size for frame in stream.frames])
            ]
        )

    def _getAVPtsContrastOption(self, v_stream_index, a_stream_index):
        v_stream = self.data.video.streams[v_stream_index]
        a_stream = self.data.audio.streams[a_stream_index]
        return dict(
            title=dict(
                text=f"pts-delta-contrast | video-{v_stream_index}-{self._getPktDurationInFrame(v_stream)}delta | "
                     f"audio-{a_stream_index}-{self._getPktDurationInFrame(a_stream)}delta"
            ),
            xAxis=[
                dict(
                    name='frame_index',
                    data=[index for index, _ in enumerate(v_stream.frames)]
                )
            ],
            yAxis=[
                dict(
                    name='pkt_pts_delta'
                )
            ],
            series=[
                dict(name="video", data=self._getPtsDiffListInFrame(v_stream)),
                dict(name="audio", data=self._getPtsDiffListInFrame(a_stream)),
            ]
        )


# http://192.168.5.5:5000/flow/info?filename=/opt/ffmpeg/sample/n230323ad109ibgaj04ow133zoew5fht.mp4&packet=0-1000
# http://192.168.5.5:5000/flow/info?filename=/opt/ffmpeg/sample/dota2/10-20.flv&time=1-3&packet=30-100
@bp.route("/info", methods=("GET",))
def info():
    if not request.args:
        abort(404, description="Request args not found")

    filename = request.args.get('filename')
    time_range = request.args.get('time')
    packet_range = request.args.get('packet')

    read_intervals = None
    if time_range:
        time_range = time_range.split('-')
        time_from = time_range[0] if int(time_range[0]) else ''
        read_intervals = f"{time_from}%+{time_range[1]}"
    if packet_range:
        packet_range = packet_range.split('-')
        packet_from = packet_range[0] if int(packet_range[0]) else ''
        if not read_intervals:
            read_intervals = f"{packet_from}%+#{packet_range[1]}"
        else:
            read_intervals += f",{packet_from}%+#{packet_range[1]}"
    if not read_intervals:
        read_intervals = "%+3"
    logging.info(f"request: {filename} {read_intervals}")

    flow = MediaInfoFlow(filename, read_intervals=read_intervals)
    return render_template("flow/info.html", **ChartOption(flow.data).getRenderData())
