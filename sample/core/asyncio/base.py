import asyncio

async def worker(q: asyncio.Queue):
    while True:
        item = await q.get()
        print("处理:", item)
        # 做一些事...
        q.task_done()
        print("unfinished_tasks =", q.unfinished_tasks)

async def main():
    q = asyncio.Queue()
    for i in range(3):
        await q.put(i)
    asyncio.create_task(worker(q))
    await q.join()
    print("队列所有任务完成")

asyncio.run(main())