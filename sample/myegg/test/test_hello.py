from unittest import TestCase

from my_egg.hello import run

class HelloTestCase(TestCase):
    def test_run(self):
        self.assertEqual(run(), 'HELLO WORLD')

