from typing import List, Optional
from pydantic import BaseModel, Field


class FFprobeFormatTag(BaseModel):
    description: Optional[str]
    metadatacreator: Optional[str]
    canSeekToEnd: Optional[bool]
    videosize: Optional[int]
    audiosize: Optional[int]
    lastkeyframetimestamp: Optional[int]
    lastkeyframelocation: Optional[int]
    encoder: Optional[str]


class FFprobeFormat(BaseModel):
    filename: str
    nb_streams: int
    nb_programs: int
    format_name: str
    format_long_name: str
    start_time: float
    duration: float
    size: int
    bit_rate: int
    probe_score: int
    tags: FFprobeFormatTag = Field(..., alias='tags')


class FFprobeVideoFrame(BaseModel):
    media_type: str
    stream_index: int
    key_frame: int
    pkt_pts: int
    pkt_pts_time: float
    pkt_dts: Optional[int]
    pkt_dts_time: Optional[float]
    best_effort_timestamp: int
    best_effort_timestamp_time: float
    pkt_duration: Optional[int]
    pkt_duration_time: Optional[float]
    pkt_pos: int
    pkt_size: float
    width: int
    height: int
    pix_fmt: str
    sample_aspect_ratio: Optional[str]
    pict_type: str
    coded_picture_number: int
    display_picture_number: int
    interlaced_frame: int
    top_field_first: int
    repeat_pict: int
    color_range: Optional[str]
    color_space: Optional[str]
    color_primaries: Optional[str]
    color_transfer: Optional[str]
    chroma_location: str


class FFprobeAudioFrame(BaseModel):
    media_type: str
    stream_index: int
    key_frame: int
    pkt_pts: int
    pkt_pts_time: float
    pkt_dts: int
    pkt_dts_time: float
    best_effort_timestamp: int
    best_effort_timestamp_time: float
    pkt_duration: int
    pkt_duration_time: float
    pkt_pos: int
    pkt_size: int
    sample_fmt: str
    nb_samples: int
    channels: int
    channel_layout: str


class FFprobeVideoPacket(BaseModel):
    codec_type: str
    stream_index: int
    pts: int
    pts_time: float
    dts: int
    dts_time: float
    duration: Optional[int]
    duration_time: Optional[float]
    convergence_duration: Optional[str]
    convergence_duration_time: Optional[str]
    size: float
    pos: int
    flags: str


class FFprobeAudioPacket(BaseModel):
    codec_type: str
    stream_index: int
    pts: int
    pts_time: float
    dts: int
    dts_time: float
    duration: int
    duration_time: float
    convergence_duration: Optional[str]
    convergence_duration_time: Optional[str]
    size: float
    pos: int
    flags: str


class FFprobeStreamDisposition(BaseModel):
    default: int
    dub: int
    original: int
    comment: int
    lyrics: int
    karaoke: int
    forced: int
    hearing_impaired: int
    visual_impaired: int
    clean_effects: int
    attached_pic: int
    timed_thumbnails: int


class FFprobeVideoStream(BaseModel):
    index: int
    codec_name: str
    codec_long_name: str
    profile: str
    codec_type: str
    codec_time_base: str
    codec_tag_string: str
    codec_tag: str
    width: int
    height: int
    coded_width: int
    coded_height: int
    has_b_frames: int
    sample_aspect_ratio: Optional[str]
    display_aspect_ratio: Optional[str]
    pix_fmt: str
    level: int
    chroma_location: str
    field_order: Optional[str]
    refs: int
    is_avc: bool
    nal_length_size: int
    r_frame_rate: str
    avg_frame_rate: str
    time_base: str
    start_pts: int
    start_time: float
    bit_rate: int
    bits_per_raw_sample: int
    disposition: FFprobeStreamDisposition = Field(..., alias='disposition')
    frames: List[FFprobeVideoFrame] = list()
    packets: List[FFprobeVideoPacket] = list()


class FFprobeAudioStream(BaseModel):
    index: int
    codec_name: str
    codec_long_name: str
    profile: str
    codec_type: str
    codec_time_base: str
    codec_tag_string: str
    codec_tag: str
    sample_fmt: str
    sample_rate: int
    channels=2
    channel_layout: str
    bits_per_sample=0
    r_frame_rate: str
    avg_frame_rate: str
    time_base: str
    start_pts: int
    start_time: float
    bit_rate: int
    disposition: FFprobeStreamDisposition = Field(..., alias='disposition')
    frames: List[FFprobeAudioFrame] = list()
    packets: List[FFprobeAudioPacket] = list()


class FFprobeVideo(BaseModel):
    streams: List[FFprobeVideoStream] = list()


class FFprobeAudio(BaseModel):
    streams: List[FFprobeAudioStream] = list()


class FFprobeData(BaseModel):
    format: Optional[FFprobeFormat]
    video: FFprobeVideo = FFprobeVideo()
    audio: FFprobeAudio = FFprobeAudio()
