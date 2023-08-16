import asyncio
import concurrent
import logging
import multiprocessing as mp
import aiohttp

logging.basicConfig(level=logging.DEBUG)


class HttpClient:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=3)
        )
        self.loop.run_until_complete(self.run())

    async def run(self):
        self.connector = aiohttp.TCPConnector(limit=1, limit_per_host=1)
        self.timeout = aiohttp.ClientTimeout(2)
        self.session = aiohttp.ClientSession(
            connector=self.connector, timeout=self.timeout
        )
        for i in range(10000):
            async with self.session.post(
                "http://mcdn-moni.net/metrics",
                headers={"X-Authorization-Pushgateway": "7d149ef552d2923f"},
            ) as resp:
                await resp.text()
                status = resp.status
                print(mp.current_process().pid, status)

            await asyncio.sleep(5)


HttpClient()
