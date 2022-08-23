import json

personDict = {'realname': 'trans-live_392836434_52698792-2022-08-21-20:53:15', 'object': {'url': 'http://uposgate-vip.bilivideo.com:2281/livechunks/trans-live_392836434_52698792-2022-08-21-20:53:15.flv', 'filesize': 221267957, 'object_id': 'trans-a7e57e20-2150-11ed-b8de-549f350efd84-116', 'md5': '76060bce27f137b6c248273bd862e247'}, 'buid': 'VOD_LPL', 'force_local': True, 'recv_uts': 1661086581.62917, 'av_samples': {}, 'av_covers': {}, 'profiles': [{'profile': '116', 'video': {'encode': {'gop_max': 300, 'height': 1080, 'bit_rate': 5836800, 'frame_rate': 60, 'bufsize': 12000000, 'x264opts': 'psy=0:', 'gop_min': 300, 'width': 1920, 'maxrate': 12000000, 'crf': 0, 'vsync': 3, 'bili_preset': 'balance', 'codec': 'h264'}, 'filter': {}}, 'audio': {'encode': {'codec': 'copy'}, 'filter': {}}, 'container': {'duration': 0, 'start_time': 0, 'segment_time': 0, 'format': 'flv'}, 'save_to': 'upos:livechunks/trans-live_392836434_52698792-2022-08-21-20:53:15'}], 'filename': 'trans-live_392836434_52698792-2022-08-21-20:53:15', 'priority': 0, 'callback': 'http://10.69.74.12/t_callback', 'role': 'fastrec'}

app_json = json.dumps(personDict, sort_keys=True)
print(app_json)

O = {'TOP': {'team1': [], 'tower1': [], 'property1': ['17.2k', 0.86696], 'score1': [], 'score2': [], 'property2': ['14.2k', 0.94939], 'tower2': [], 'team2': [], 'time': ['05:49', 0.96117], 'event': ['123', 0.9258], 'killee': ['444', 0.9093374446523449]}}
event = O["TOP"].get("event", [])
killee = O["TOP"].get("killee", [])
print(event, killee)
if event or killee:
    result = dict(event = event, killee = killee)
    print(result)

    target = None
    event = result.get("event")
    if event and len(event) == 2:
        target = event
    print(target)
    killee = result.get("killee")
    if killee and len(killee) == 2:
        if not target:
            target = killee
            print(target)
        else:
            target[0] += "-" + killee[0]
            target[1] = float(event[1] + killee[1]) / 2
    print(target)

