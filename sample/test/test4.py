import aiofiles
import aiohttp
import asyncio
import time


async def task_1():
    async with aiohttp.ClientSession() as session:
        print(f"111->{session}")
        async with session.get('http://python.org') as response:
            print("Status:", response.status)
            print("Content-type:", response.headers['content-type'])
            html = await response.text()
            print("Body:", html[:15], "...")

async def task_2():
    async with aiohttp.ClientSession() as session:
        print(f"222->{session}")
        async with session.get('http://bilibilitest-1252693259.cosgz.myqcloud.com/2022-01-20-live_867152_6880638/20220120113530.png') as response:
            print("Status:", response.status)
            if response.status == 200:
                print("Content-type:", response.headers['content-type'])
                async with aiofiles.open('20220120113530-0.png', mode='wb') as f:
                    await f.write(await response.read())

async def task_3():
    async with aiohttp.ClientSession() as session:
        print(f"333->{session}")
        async with session.get('http://bilibilitest-1252693259.cosgz.myqcloud.com/2022-01-20-live_867152_6880638/20220120113530.png') as response:
            print("Status:", response.status)
            if response.status == 200:
                print("Content-type:", response.headers['content-type'])
                f = await aiofiles.open('20220120113530-1.png', mode='wb')
                await f.write(await response.read())
                await f.close()

async def task_4():
    async with aiohttp.ClientSession() as session:
        print(f"444->{session}")
        async with session.get('http://bilibilitest-1252693259.cosgz.myqcloud.com/2022-01-20-live_867152_6880638/20220120113530.png') as response:
            print("Status:", response.status)
            if response.status == 200:
                print("Content-type:", response.headers['content-type'])
                f = await aiofiles.open('20220120113530-2.png', mode='wb')
                await f.write(await response.read())
                await f.close()
        return 4

async def main():
    while True:
        L = await asyncio.gather(
            task_1(),
            task_2(),
            task_3(),
            task_4(),
            task_4(),
        )
        print(L)
        break

print("ENTER")
loop = asyncio.get_event_loop()
#loop.run_until_complete(main())
asyncio.run(main())
print("FINISH")