import os
from pydantic import BaseSettings
from util.basic import version_from_readme


class _Settings(BaseSettings):
    name: str = 'av-tool'
    version: str = version_from_readme(None)
    dir_conf: str = os.path.dirname(os.path.realpath(__file__))
    dir_source: str = os.path.dirname(dir_conf)
    dir_project: str = os.path.dirname(dir_source)
    dir_workspace: str = f"{dir_project}/work"
    dir_log: str = f"{dir_workspace}/{name}.log"

    bin_ffprobe: str = f"/opt/ffmpeg/bin/ffprobe -hide_banner"

    class Config:
        env_prefix = "av_"
        env_file = "env.py"


setting = _Settings()
del _Settings
