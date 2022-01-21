# -*- coding: utf-8 -*-
import os
import sys
import logging
import aiohttp
import aiofiles
import asyncio
import queue
import time
import threading
from urllib.parse import urlparse


def init_local_log(logging_level=logging.INFO):
    fmt = '[%(levelname)1.1s %(asctime)s %(module)-16.16s:%(lineno)4d] %(message)s'
    date_fmt = '%y%m%d %H:%M:%S'
    logging.basicConfig(format=fmt, datefmt=date_fmt, level=logging_level)

init_local_log()


class AsyncIO:

    def __init__(self, maxsize=10000, retries=0):
        self.retries = retries
        self.loop = asyncio.get_event_loop()
        self.from_q = queue.Queue(maxsize)
        self.to_q = queue.Queue(maxsize)
        self.worker = threading.Thread(target=self._run, daemon=True)

    def start(self):
        logging.info(f"start")
        self.worker.start()

    def stop(self):
        logging.info(f"stop")
        self.put({'exit': True})
        self.worker.join()

    def put(self, req):
        if not req:
            return None
        self.from_q.put(req)

    def get(self):
        if self.to_q.empty():
            return None
        return self.to_q.get()

    def _run(self):
        async def main():
            index = 0
            tasks = {}
            exit = False
            while True:
                def get_task():
                    nonlocal index, tasks, exit
                    item = self.from_q.get()
                    logging.info(f"item: {item}")
                    exit = item.get('exit') if item else None
                    if exit:
                        return None
                    url = item.get('url') if item else None
                    path = item.get('path') if item else None
                    retry = 0
                    task = asyncio.ensure_future(self._run_http(index, retry, url, path))
                    tasks[index] = {
                        'url': url,
                        'retry': retry,
                        'task': task,
                    }
                    index += 1
                    return task

                def del_task(index):
                    nonlocal tasks
                    self.from_q.task_done()
                    tasks.pop(index)

                while not self.from_q.empty():
                    if not get_task():
                        exit = True
                        break

                if exit:
                    logging.info(f"exit")
                    break

                # block when queue is empty
                if not tasks:
                    if not get_task():
                        logging.info(f"exit when empty")
                        break

                tasklist = [item['task'] for item in tasks.values()]
                logging.info(f"index: {index} tasks: {len(tasklist)}")
                results = await asyncio.gather(*tasklist, return_exceptions=True)
                logging.info(f"index: {index} results: {results}")

                for idx, retry, succ in results:
                    if succ:
                        del_task(idx)
                    else:
                        retry = tasks[idx]['retry']
                        retry += 1
                        if retry > retries:
                            del_task(idx)

        self.loop.run_until_complete(main())

    async def _run_http(self, index, retry, url, path):
        code = 255
        if url:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    code = resp.status
                    if code == 200:
                        path = path if path else os.path.basename(url)
                        async with aiofiles.open(path, mode='wb') as f:
                            await f.write(await resp.read())
                            code = 0

        self.to_q.put({
            "code": code,
            "url": url,
            "path": path,
        })
        return index, retry, code == 0


class TestAsyncIO:
    def __init__(self):
        self.aio = AsyncIO()
        self.urls = [
            #"https://mirrors.tuna.tsinghua.edu.cn/debian/dists/buster/main/Contents-arm64.gz",
            #"https://mirrors.tuna.tsinghua.edu.cn/debian/dists/buster/main/Contents-i386.gz",
            #"https://mirrors.tuna.tsinghua.edu.cn/debian/dists/buster/main/Contents-amd64.gz",
            #"http://bilibilitest-1252693259.cosgz.myqcloud.com/2022-01-20-live_867152_6880638/20220120113530.png",
            "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fup.enterdesk.com%2Fedpic%2F6e%2F7e%2Fdd%2F6e7edd30f22f2b136a696d885bc2f7bd.jpg&refer=http%3A%2F%2Fup.enterdesk.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1645322947&t=ffd2ca835d618fd32ea635481db9361f",
        ] * 1

    async def _run(self):
        tasks = []
        for url in self.urls:
            task = asyncio.ensure_future(self.aio._run_http(0, 0, url))
            tasks.append(task)
        result = await asyncio.gather(*tasks, return_exceptions=True)
        logging.info(f"11test: {result}")

    def run(self):
        self.aio.start()
        while True:
            for url in self.urls:
                #logging.info(f"url: {url}")
                self.aio.put({'url': url})
            time.sleep(2)
        #self.aio.stop()


test = TestAsyncIO()
test.run()
time.sleep(10)
logging.info(f"End.")
