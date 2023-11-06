from typing import List, Dict


class VXCodeRoute(object):
    PREFER_GPU = False
    ALLOW_EDGE = True
    IS_PRIVILEGED = False
    ROLE = ""
    HIGH_IO = False
    USERS_PRE_DEFAULT = ["all", "pre"]
    USERS_RED_CHANNEL = ["vxcode_high_priority", "vxcode_red_channel"]

    EXTERNAL_URLS = {
        "bos": "10.227.0.13",
        "kodo": "qiniucdn.com",
        "ks3": "ksyun.com",
        "cos": "myqcloud.com",
    }

    @classmethod
    def is_vendor_stored(cls, url):
        for i in cls.EXTERNAL_URLS.values():
            if i in url:
                return True
        else:
            return False

    @classmethod
    def is_gpu_profiles(cls, profiles: List[Dict]) -> bool:
        if profiles and profiles[0].get("arch") == "gpu":
            return True
        else:
            return False

    @classmethod
    def route(cls, task_request, env=""):
        task_buid = task_request["buid"]
        red_channel = task_request.get("red_channel", False)
        info = {
            "users": [],
        }
        # pre环境新增一个 {"users":  ["all", "pre"]}
        if env == "pre":
            info["users"] += cls.USERS_PRE_DEFAULT
        if red_channel:
            info["users"] += cls.USERS_RED_CHANNEL
        if "VJT" in task_buid:
            info["is_privileged"] = True
        if "GR:STREAM" in task_buid:
            info["gpu_type"] = "xcode"
            info["users"] += ["gpu_t4"]
        return info


print(VXCodeRoute.route({"buid": "GR:STREAM"}))
print(VXCodeRoute.route({"buid": "test"}))
print(VXCodeRoute.route({"buid": "test", "red_channel": True}))
print(VXCodeRoute.route({"buid": "GR:STREAM"}))

print(VXCodeRoute.route({"buid": "GR:STREAM"}, "pre"))
print(VXCodeRoute.route({"buid": "test"}, "pre"))
print(VXCodeRoute.route({"buid": "test", "red_channel": True}, "pre"))
print(VXCodeRoute.route({"buid": "GR:STREAM"}, "pre"))

print(VXCodeRoute.route({"buid": "GR:STREAM"}, "pre"))
print(VXCodeRoute.route({"buid": "test"}, "pre"))
print(VXCodeRoute.route({"buid": "test", "red_channel": True}, "pre"))
print(VXCodeRoute.route({"buid": "GR:STREAM"}, "pre"))
