import json

personDict = {'realname': 'trans-live_392836434_52698792-2022-08-21-20:53:15', 'object': {'url': 'http://uposgate-vip.bilivideo.com:2281/livechunks/trans-live_392836434_52698792-2022-08-21-20:53:15.flv', 'filesize': 221267957, 'object_id': 'trans-a7e57e20-2150-11ed-b8de-549f350efd84-116', 'md5': '76060bce27f137b6c248273bd862e247'}, 'buid': 'VOD_LPL', 'force_local': True, 'recv_uts': 1661086581.62917, 'av_samples': {}, 'av_covers': {}, 'profiles': [{'profile': '116', 'video': {'encode': {'gop_max': 300, 'height': 1080, 'bit_rate': 5836800, 'frame_rate': 60, 'bufsize': 12000000, 'x264opts': 'psy=0:', 'gop_min': 300, 'width': 1920, 'maxrate': 12000000, 'crf': 0, 'vsync': 3, 'bili_preset': 'balance', 'codec': 'h264'}, 'filter': {}}, 'audio': {'encode': {'codec': 'copy'}, 'filter': {}}, 'container': {'duration': 0, 'start_time': 0, 'segment_time': 0, 'format': 'flv'}, 'save_to': 'upos:livechunks/trans-live_392836434_52698792-2022-08-21-20:53:15'}], 'filename': 'trans-live_392836434_52698792-2022-08-21-20:53:15', 'priority': 0, 'callback': 'http://10.69.74.12/t_callback', 'role': 'fastrec'}

app_json = json.dumps(personDict, sort_keys=True)
print(app_json)

class T:
    DEFAULT = [1, 2]
    def get(self, *v):
        r = self.DEFAULT
        r += v
        return r

t = T()
print(t.get(3)) #[1, 2, 3]
print(t.get(3)) #[1, 2, 3, 3]
print(t.get(3)) #[1, 2, 3, 3, 3]
