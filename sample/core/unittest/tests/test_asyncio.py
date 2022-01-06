import unittest
from unittest import IsolatedAsyncioTestCase

events = []


class Test(IsolatedAsyncioTestCase):
    def setUp(self):
        events.append("setUp")

    async def asyncSetUp(self):
        #self._async_connection = await AsyncConnection()
        events.append("asyncSetUp")

    async def test_response(self):
        events.append("test_response")
        #rsp.status_code = await self._async_connection.get("https://example.com")
        #self.assertEqual(rsp.status_code, 200)
        self.addAsyncCleanup(self.on_cleanup)

    def tearDown(self):
        events.append("tearDown")

    async def asyncTearDown(self):
        #await self._async_connection.close()
        events.append("asyncTearDown")

    async def on_cleanup(self):
        events.append("cleanup")
        print(f"{events}")


if __name__ == "__main__":
    unittest.main(argv=[123], exit=True, verbosity=1)
