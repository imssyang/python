
from a import job, RequestModel

def change():
    # 修改全局 job 对象的 req 字段
    job.req = RequestModel(foo="from_b")

