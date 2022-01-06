
import logging
from subprocess import Popen, PIPE, STDOUT

with Popen(['/data/app/vmaf-worker/assets/bin/upclone', '--config', '/data/app/vmaf-worker/conf/upos/dev/upclone.conf', 'copyto', 'upos://sucai/2m4aolhkojmvfb5ed4l1587112578964.mp4', '/data/bili_vxcode_workspace/vx202112130000000000000000000264_1_origin.dat.tmp', '--upos-endpoint', 'http://upos-hz-uat.bilivideo.com', '--upos-extra-query', ''], stdout=PIPE, stderr=STDOUT, shell=True) as process:
    try:
        stdout, stderr = process.communicate(timeout=10)
    except TimeoutExpired:
        print("TimeoutExpired")
        process.kill()
        stdout, stderr = process.communicate()
    except:
        print("OtherException")
        process.kill()
        raise
    finally:
        retcode = process.poll()
    print(f"retcode: {retcode} stdout: {stdout} stderr: {stderr}")
