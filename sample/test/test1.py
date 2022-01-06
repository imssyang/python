
from subprocess import Popen, PIPE, STDOUT

p = Popen(['ls', '/home'], stdout=PIPE, stderr=STDOUT, shell=True)
print(p.communicate())
p.stdout.close()
p.wait()
