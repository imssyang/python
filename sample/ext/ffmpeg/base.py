import ffmpeg


def base():
    stream = ffmpeg.input("input.mp4")
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, "output.mp4")
    ffmpeg.run(stream)


def FlipVideoHorizont():
    ffmpeg.input("input.mp4").hflip().output("output.mp4").run()


def FpsFilter():
    (
        ffmpeg.input("dummy.mp4")
        .filter("fps", fps=25, round="up")
        .output("dummy2.mp4")
        .run()
    )


def MultipleInputFilter():
    main = ffmpeg.input("main.mp4")
    logo = ffmpeg.input("logo.png")
    (ffmpeg.filter([main, logo], "overlay", 10, 10).output("out.mp4").run())


def MultipleOutputFilter():
    split = ffmpeg.input("in.mp4").filter_multi_output("split")  # or `.split()`
    print(split)
    cmd = ffmpeg.concat(split[0], split[1]).output("out.mp4").compile()
    print(cmd)
    # ['ffmpeg', '-i', 'in.mp4', '-filter_complex', '[0]split=2[s0][s1];[s0][s1]concat=n=2[s2]', '-map', '[s2]', 'out.mp4']


MultipleOutputFilter()


def StringExpressionFilter():
    args = (
        ffmpeg.input("in.mp4")
        .filter("crop", "in_w-2*10", "in_h-2*20")
        .output("out.mp4")
        .get_args()
    )
    print(args)
    # ['-i', 'in.mp4', '-filter_complex', '[0]crop=in_w-2*10:in_h-2*20[s0]', '-map', '[s0]', 'out.mp4']


StringExpressionFilter()


def SpecialOption():
    args = ffmpeg.input("in.mp4").output("out.mp4", **{"qscale:v": 3}).get_args()
    print(args)
    """['-i', 'in.mp4', '-qscale:v', '3', 'out.mp4']"""


SpecialOption()


def ComplexFilter():
    in_file = ffmpeg.input("input.mp4", **{"ss": 20})
    overlay_file = ffmpeg.input("overlay.png")
    args = (
        ffmpeg.concat(
            in_file.trim(start_frame=10, end_frame=20),
            in_file.trim(start_frame=30, end_frame=40),
        )
        .overlay(overlay_file.hflip())
        .drawbox(50, 50, 120, 120, color="red", thickness=5)
        .output("out.mp4")
        .get_args()
    )
    print(args)
    """
    ['-i', 'input.mp4',
    '-i', 'overlay.png',
    '-filter_complex',
    '[0]trim=end_frame=20:start_frame=10[s0];
     [0]trim=end_frame=40:start_frame=30[s1];
     [s0][s1]concat=n=2[s2];
     [1]hflip[s3];
     [s2][s3]overlay=eof_action=repeat[s4];
     [s4]drawbox=50:50:120:120:red:t=5[s5]',
    '-map',
    '[s5]',
    'out.mp4']
    """


def ComplexFilter2():
    in_file = ffmpeg.input("input.mp4")
    overlay_file = ffmpeg.input("overlay.png")
    args = (
        ffmpeg.concat(
            in_file.trim(start_frame=10, end_frame=20),
            in_file.trim(start_frame=30, end_frame=40),
        )
        .overlay(overlay_file.hflip())
        .drawbox(50, 50, 120, 120, color="red", thickness=5)
        .output("out.mp4")
        .view(filename="filter_graph", detail=True)
        .run()
    )
    print(args)


# MultipleOutputFilter()
ComplexFilter()
