import json


def dict_to_json():
    d = {
        "realname": "trans-live_392836434_52698792-2022-11-01-18:47:06",
        "object": {
            "url": "http://uposgate-vip.bilivideo.com:2281/livechunks/trans-live_392836434_52698792-2022-11-01-18:47:06.flv",
            "filesize": 232760928,
            "object_id": "trans-f62b6efe-59d2-11ed-84bc-549f350efd84-64",
            "md5": "4aa3a977e51b281dfcc29fa4e02e11d3",
        },
        "buid": "VOD_LPL",
        "force_local": True,
        "recv_uts": 1667299812.640104,
        "av_samples": {},
        "av_covers": {},
        "profiles": [
            {
                "profile": "64",
                "video": {
                    "encode": {
                        "gop_max": 150,
                        "height": 720,
                        "bit_rate": 1945600,
                        "frame_rate": 30,
                        "bufsize": 12000000,
                        "x264opts": "psy=0:",
                        "gop_min": 150,
                        "width": 1280,
                        "maxrate": 12000000,
                        "crf": 25,
                        "vsync": 3,
                        "bili_preset": "auto_crf",
                        "codec": "h264",
                    },
                    "filter": {},
                    "based": "64",
                },
                "audio": {"encode": {"codec": "copy"}, "filter": {}},
                "container": {
                    "duration": 0,
                    "start_time": 0,
                    "segment_time": 0,
                    "format": "flv",
                },
                "save_to": "upos:livechunks/trans-live_392836434_52698792-2022-11-01-18:47:06",
            }
        ],
        "filename": "trans-live_392836434_52698792-2022-11-01-18:47:06",
        "priority": 0,
        "callback": "http://10.69.74.12/t_callback",
        "role": "fastrec",
    }
    j = json.dumps(d, indent=4)
    print(j)
    k = json.dumps(j)
    print(k)
    k = json.dumps(k)
    print(k)
    o = json.loads(k)
    print(o)



def json_to_dict():
    j = '{"id": "007", "name": "007", "age": 28, "sex": "male", "phone": "13000000000", "email": "123@qq.com"}'
    d = json.loads(s=j)
    print(d)  # {'id': '007', 'name': '007', 'age': 28, 'sex': 'male', 'phone': '13000000000', 'email': '123@qq.com'}


if __name__ == "__main__":
    dict_to_json()
