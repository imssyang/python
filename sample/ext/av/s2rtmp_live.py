import av
import logging
import sys

# -----------------------------
# 配置 Python logging
# -----------------------------
logger = logging.getLogger("pyav_demo")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -----------------------------
# 配置 PyAV / FFmpeg 日志
# -----------------------------
# av.logging.set_level(level)
#   可选级别: "quiet", "panic", "fatal", "error", "warning", "info", "verbose", "debug", "trace"
av.logging.set_level(av.logging.INFO)

# -----------------------------
# PyAV RTMP 转码示例
# -----------------------------
input_url = "rtmp://218.84.107.136/live/livestream"
output_url = "rtmp://localhost/live/livestream"

# 打开输入 RTMP 流
logger.info(f"Opening input stream: {input_url}")
input_container = av.open(input_url, format='flv', mode='r')

# 打开输出 RTMP 流
logger.info(f"Opening output stream: {output_url}")
output_container = av.open(
    output_url, format='flv', mode='w'
)

# 找到视频和音频流
input_video = next(s for s in input_container.streams if s.type == 'video')
input_audio = next((s for s in input_container.streams if s.type == 'audio'), None)

# 添加输出流（转码为 H.264 / AAC）
out_video = output_container.add_stream('libx264', rate=input_video.average_rate)
out_video.width = input_video.width
out_video.height = input_video.height
out_video.pix_fmt = 'yuv420p'

if input_audio:
    out_audio = output_container.add_stream('aac', rate=input_audio.rate)
else:
    out_audio = None

# 开始转码循环
logger.info("Start transcoding loop...")
try:
    for packet in input_container.demux(input_video, input_audio):
        logger.info(f"Processing packet from stream {packet.stream.index}:{packet.stream.type} pts={packet.pts} dts={packet.dts}")
        for frame in packet.decode():
            if packet.stream.type == 'video':
                frame = frame.reformat(width=input_video.width, height=input_video.height, format='yuv420p')
                for packet_out in out_video.encode(frame):
                    output_container.mux(packet_out)
            elif packet.stream.type == 'audio' and out_audio:
                for packet_out in out_audio.encode(frame):
                    output_container.mux(packet_out)
except Exception as e:
    logger.exception(f"Error during transcoding: {e}")

# flush 剩余数据
for packet in out_video.encode():
    output_container.mux(packet)

if out_audio:
    for packet in out_audio.encode():
        output_container.mux(packet)

input_container.close()
output_container.close()
logger.info("Transcoding completed.")
