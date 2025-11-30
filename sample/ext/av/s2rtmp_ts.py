import av
import logging
import sys
import time

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
av.logging.set_level(av.logging.INFO)

# -----------------------------
# PyAV RTMP 转码示例
# -----------------------------
input_url = "rtmp://218.84.107.136/live/livestream"
output_file = "/opt/python/sample/ext/av/s2rtmp.ts"

logger.info(f"Opening input stream: {input_url}")
try:
    input_container = av.open(input_url, format="flv", mode="r", timeout=5.0)
except av.AVError as e:
    logger.error(f"无法打开输入流: {e}")
    sys.exit(1)

logger.info(f"Creating output file: {output_file}")
output_container = av.open(output_file, format='mpegts', mode='w')

# 找到视频和音频流
input_video = next((s for s in input_container.streams if s.type == 'video'), None)
input_audio = next((s for s in input_container.streams if s.type == 'audio'), None)
if not input_video:
    logger.error("没有检测到视频流")
    sys.exit(1)

# 添加输出流（转码为 H.264 / AAC）
out_video = output_container.add_stream('libx264', rate=input_video.average_rate or 25)
out_video.width = input_video.width
out_video.height = input_video.height
out_video.pix_fmt = 'yuv420p'

out_audio = None
if input_audio:
    out_audio = output_container.add_stream("aac", rate=input_audio.rate)
    out_audio.channels = input_audio.channels
    out_audio.layout = input_audio.layout

# 开始转码循环
logger.info("Start transcoding loop...")
last_packet_time = time.time()
timeout_seconds = 10  # 输入断流超时判定（秒）

try:
    for packet in input_container.demux(input_video, input_audio):
        logger.info(f"Processing packet from stream {packet.stream.index}:{packet.stream.type} pts={packet.pts} dts={packet.dts}")

        # 断流检测
        now = time.time()
        if now - last_packet_time > timeout_seconds:
            logger.warning("输入流超时未接收数据，自动退出转码")
            break

        if packet.dts is None:
            continue  # 跳过无效帧

        last_packet_time = now

        for frame in packet.decode():
            if packet.stream.type == 'video':
                frame = frame.reformat(width=input_video.width, height=input_video.height, format='yuv420p')
                for packet_out in out_video.encode(frame):
                    output_container.mux(packet_out)
            elif packet.stream.type == 'audio' and out_audio:
                for packet_out in out_audio.encode(frame):
                    output_container.mux(packet_out)
except av.error.ExitError as e:
    logger.warning(f"RTMP input is interrupted: {e}")
except av.EOFError:
    logger.info("The input stream ends, stopping transcoding.")
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
