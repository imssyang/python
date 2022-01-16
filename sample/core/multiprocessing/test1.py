from multiprocessing import Pool


def part_crack_helper(args):
    solution = True
    if solution:
        return True
    else:
        return False


class Worker():
    def __init__(self, workers, initializer, initargs):
        self.pool = Pool(processes=workers,
                         initializer=initializer,
                         initargs=initargs)

    def callback(self, result):
        if result:
            print("Solution found! Yay!")
            self.pool.terminate()

    def do_job(self):
        self.pool.apply_async(part_crack_helper,
                              args=[],
                              callback=self.callback)

        self.pool.close()
        self.pool.join()
        print("good bye")


w = Worker(1, None, None)
w.do_job()
