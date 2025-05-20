
from job import JobModel

if __name__ == "__main__":
    data = {'buid': 'Main:UGC_DASH:NBv1',
        'callback': 'http://uat-vxcode-streaming-scheduler.bilibili.co/api/callback?job_id=sj202412230000000000000001271629&bayes=true',
        'cid': 12802459148,
        'dry_run': False,
        'flow': 'probe',
        'priority': 0,
        'request': {'file': {'md5': 'unknown',
                            'size': 1438798,
                            'url': 'lfos://stcode/cae16721e03d4c709c275e1ff2dcb8d6d59c9/1241223wnpeliz8l1b3yp269li4kfz4k.m3u8'}},
        'resource': {'CPU': {'optional_cpu': 6000,
                            'optional_mem': 8192,
                            'optional_storage': 4000,
                            'required_cpu': 2000,
                            'required_gpu': 0,
                            'required_mem': 2048,
                            'required_storage': 1000}},
        'response': {},
        'submission_id': '12345',
        'task_id': 'sp202412230000000000000001271620'}

    print(type(data))
    job = JobModel(**data)
    print(job)


