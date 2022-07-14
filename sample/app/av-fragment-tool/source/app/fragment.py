# -*- coding: utf-8 -*-
import time
import traceback
from datetime import datetime
from utils import VXCodeUtility


class FragmentData:
    def __init__(self, room: str, extra: int, fragments: list, dataset: list):
        self.room = room
        self.extra = extra
        self.fragments = fragments
        self.dataset = dataset

    def truncate_all(self):
        for fragment in self.fragments:
            data1 = data2 = dict()
            for data in self.dataset:
                from_time = data["from"].timestamp()
                to_time = data["to"].timestamp()
                if (from_time <= fragment[0] and fragment[0] <= to_time) or (
                    from_time <= fragment[1] and fragment[1] <= to_time
                ):
                    if not data1:
                        data1 = data
                    elif not data2:
                        data2 = data
            if data1:
                self._truncate(fragment, data1, data2)

    def _truncate(self, fragment, data1, data2):
        print(f"[FIND]: {fragment} in {data1}")
        path1 = data1["path"]
        fragment_start = fragment[0] - self.extra
        fragment_end = fragment[1] + self.extra
        ss_param = fragment_start - data1["from"].timestamp()
        ss_param = ss_param if 0 < ss_param else 0
        to_param = fragment_end - data1["from"].timestamp()
        from_time = datetime.fromtimestamp(fragment_start)
        from_time_text = from_time.strftime("%Y%m%d-%H%M%S")
        to_time = datetime.fromtimestamp(fragment_end)
        to_time_text = to_time.strftime("%H%M%S")
        cmd = (
            f"assets/bin/ffmpeg4 -y -nostdin -hide_banner -v info -probesize 400000000 "
            f"-i {path1} "
            f"-ss {ss_param} -to {to_param} -c copy out/{self.room}_{from_time_text}_{to_time_text}.flv"
        )

        try:
            print(f"[FRAGMENT]: {cmd}")
            result = VXCodeUtility.exec_subprocess(cmd)
            if not result:
                print(f"cmd failed with: {result}")
        except Exception as e:
            print(f"ffmpeg exception={e} traceback={traceback.format_exc()}")
