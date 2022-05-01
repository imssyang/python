# -*- coding: utf-8 -*-
import os
import logging
import aiohttp
import aiofiles
import asyncio
import queue
import signal
import threading
import time
import multiprocessing as mp
import concurrent.futures as cf
from urllib.parse import urlparse
from utils import LOCAL_PATH
from utils import link_file
from utils import current_process_info
from utils import current_process_and_thread


class AsyncIO:
    def __init__(
        self, queue_size=10000, pool_size=1, workers=10, timeout=30, retries=0
    ):
        signal.signal(signal.SIGCHLD, self._signal_handler)
        self.ERROR_CODE_START = 600
        self.pool_size = pool_size
        self.workers = workers
        self.executor = None
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.queue_size = queue_size
        self.retries = retries
        self.loop = {}
        self.from_q = mp.Queue(queue_size)
        self.to_q = mp.Queue(queue_size)
        self.exit_q = mp.Queue(pool_size * workers)
        self.process_pool = {}
        for i in range(pool_size):
            self.process_pool[i] = mp.Process(
                target=self._run_executor, args=(i,), name="vxcode-process-asyncio"
            )
        logging.info(
            f"queue_size: {queue_size} pool_size: {pool_size} "
            f"workers: {workers} timeout: {timeout} retries: {retries}"
        )

    def start(self):
        for i in range(self.pool_size):
            self.process_pool[i].start()
        logging.info(f"start process pool: {self.pool_size}")

    def stop(self, timeout):
        total = 0
        while True:
            if total > timeout:
                logging.info(f"break when timeout: {timeout}")
                break

            if self.exit_q.full():
                break

            self.put({"exit": True})
            time.sleep(0.1)
            total += 0.1

        for i in range(self.pool_size):
            self.process_pool[i].join(timeout)
        logging.info(
            f"stop process pool: {self.pool_size} exit_q_size: {self.exit_q.qsize()} after sleep: {total}"
        )

    def put(self, req):
        if not req:
            return None
        self.from_q.put(req)

    def get(self, timeout=None):
        return self.to_q.get(True, timeout)

    def _signal_handler(self, signum, frame):
        cpid, status = os.waitpid(-1, os.WNOHANG)
        logging.info(f"childpid: {cpid} exit status: {status}")

    def _run_executor(self, *args):
        (eid,) = args
        logging.info(f"start executor: {current_process_info()} eid: {eid}")
        with cf.ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="vxcode-thread-asyncio"
        ) as self.executor:
            self.executor.map(
                self._run_worker, [eid * self.workers + i for i in range(self.workers)]
            )
        logging.info(f"stop executor: {current_process_info()} eid: {eid}")

    def _run_worker(self, *args):
        (wid,) = args

        async def main(wid):
            logging.info(f"enter worker-main[{wid}]: {current_process_and_thread()}")
            index = 0
            tasks = {}
            exit = False
            code = self.ERROR_CODE_START
            detail = "fail"

            while True:

                def get_task():
                    nonlocal wid, index, tasks, exit
                    item = self.from_q.get()
                    exit = item.get("exit") if item else None
                    if exit:
                        return None

                    if not item:
                        code = self.ERROR_CODE_START + 1
                        detail = "not found item"
                        self._put_result(code, detail)
                        return code

                    url = item.get("url")
                    if not url:
                        code = self.ERROR_CODE_START + 2
                        detail = "not found url"
                        self._put_result(code, detail)
                        return code

                    scheme = urlparse(url).scheme
                    method_name = f"_run_{scheme}"
                    if method_name not in dir(self):
                        code = self.ERROR_CODE_START + 3
                        detail = f"unsupport scheme {url}"
                        self._put_result(code, detail)
                        return code

                    run_method = getattr(self, method_name)
                    if not callable(run_method):
                        code = self.ERROR_CODE_START + 4
                        detail = f"can't call {method_name}"
                        self._put_result(code, detail)
                        return code

                    path = item.get("path")
                    if not path:
                        subdir = item.get("subdir")
                        name = str(wid) + "_" + str(index) + "_" + os.path.basename(url)
                        path = LOCAL_PATH(subdir, name)
                        item.update({"path": path})

                    retry = 0
                    task = asyncio.ensure_future(
                        run_method(wid, index, retry, url, path, item)
                    )
                    tasks[index] = {
                        "retry": retry,
                        "task": task,
                    }
                    index += 1
                    return 0

                def del_task(index):
                    nonlocal tasks
                    tasks.pop(index)

                while not self.from_q.empty():
                    code = get_task()
                    if code is None:
                        exit = True
                        break

                if exit:
                    logging.info(f"exit")
                    self.exit_q.put({"pid": mp.current_process().pid, "wid": wid})
                    break

                # block when queue is empty
                if not tasks:
                    code = get_task()
                    if code is None:
                        logging.info(f"exit when empty")
                        self.exit_q.put({"pid": mp.current_process().pid, "wid": wid})
                        break
                    elif code != 0:
                        continue

                tasklist = [item["task"] for item in tasks.values()]
                results = await asyncio.gather(*tasklist, return_exceptions=False)
                from_qsize = self.from_q.qsize()
                to_qsize = self.to_q.qsize()
                if (
                    abs(self.queue_size - from_qsize) < 100
                    or abs(self.queue_size - to_qsize) < 100
                ):
                    logging.warn(
                        f"from_qsize: {from_qsize} to_qsize: {to_qsize} "
                        f"tasks: {len(tasklist)} results: {len(results)}"
                    )

                try:
                    for wid, idx, retry, succ in results:
                        if succ:
                            del_task(idx)
                        else:
                            retry = tasks[idx]["retry"]
                            retry += 1
                            if retry > self.retries:
                                del_task(idx)
                except Exception as e:
                    logging.info(f"exception: {e}")

        self.loop[wid] = asyncio.new_event_loop()
        self.loop[wid].run_until_complete(main(wid))
        logging.info(f"exit worker[{wid}]: {current_process_and_thread()}")

    def _put_result(self, code, detail, item=None):
        result = {
            "code": code,
            "detail": detail,
        }
        if isinstance(item, dict):
            result.update(item)
        self.to_q.put(result)

    async def _run_https(self, wid, index, retry, url, path, item):
        return await self._run_http(wid, index, retry, url, path, item)

    async def _run_http(self, wid, index, retry, url, path, item):
        code = self.ERROR_CODE_START + 10
        detail = "http fail"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as resp:
                    code = resp.status
                    if code == 200:
                        logging.info(f"Download {url} -> {path}")
                        async with aiofiles.open(path, mode="wb") as f:
                            await f.write(await resp.read())
                            code = 0
                            detail = "success"
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            code = self.ERROR_CODE_START + 11
            detail = e

        self._put_result(code, detail, item)
        return wid, index, retry, code == 0

    async def _run_file(self, wid, index, retry, url, path, item):
        code = self.ERROR_CODE_START + 20
        detail = "file fail"

        try:
            src_path = url.replace("file://", "")
            result = link_file(src_path, path)
            code = result["retcode"]
            detail = result["stdout"]
        except Exception as e:
            code = self.ERROR_CODE_START + 21
            detail = e

        self._put_result(code, detail, item)
        return wid, index, retry, code == 0
