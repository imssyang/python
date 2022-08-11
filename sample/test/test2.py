import copy

class VXCodeRoute:
    PREFER_GPU = False
    ALLOW_EDGE = True
    IS_PRIVILEGED = False
    ROLE = ""
    HIGH_IO = False
    USERS_PRE_DEFAULT = ["all", "pre"]
    USERS_RED_CHANNEL = ["vxcode_high_priority", "vxcode_red_channel"]

    @classmethod
    def route(cls, task_request, has_hdfs=False):
        task_object = task_request["object"]
        object_url = task_object["url"]
        task_priority = task_request["priority"]
        red_channel = task_request.get("red_channel", False)
        info = {}
        if True:
            info["users"] = copy.copy(cls.USERS_PRE_DEFAULT)
        if red_channel:
            info["users"] += cls.USERS_RED_CHANNEL
        if has_hdfs:
            info["allow_edge"] = False
        print(f"{info}")
        return info


for i in range(10):
    VXCodeRoute.route({'red_channel': True, 'object': {'url': '111'}, 'priority': 1})
    print(f"{VXCodeRoute.USERS_PRE_DEFAULT}")


