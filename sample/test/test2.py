
import logging
from subprocess import Popen, PIPE, STDOUT

with Popen(['ls', '/home'], stdout=PIPE, stderr=STDOUT, shell=True) as process:
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
