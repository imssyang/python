from unittest import TestCase

from core.unittest.app.hello import run

class HelloTestCase(TestCase):
    def test_run(self):
        self.assertEqual(run(), 'HELLO WORLD')
