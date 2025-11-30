import av

av.logging.set_level(av.logging.VERBOSE)
container = av.open("/opt/ffmpeg/sample/mp4/bbb_640x360_60fps_1200k.mp4")

for index, frame in enumerate(container.decode(video=0)):
    frame.to_image().save(f"frame-{index:04d}.jpg")
