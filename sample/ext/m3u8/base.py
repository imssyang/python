import os
import m3u8


def get_vp_m3u8(path, uries):
    m = m3u8.load(path)
    vp_segments = m3u8.SegmentList()
    for segment in m.segments:
        if segment.uri in uries:
            vp_segments.append(segment)
            print(segment.uri)

    m.segments = vp_segments
    base, ext = os.path.splitext(path)
    vp_path = f"{base}.vp{ext}"
    m.dump(vp_path)


uries = [
    "output6.m4s",
    "output7.m4s",
    "output88.m4s",
    "output9.m4s",
    "output10.m4s",
    "output11.m4s",
]
get_vp_m3u8("output.m3u8", uries)
