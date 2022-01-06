import unittest
from app.widget import Widget

def setUpModule():
    print("setUpModule %s" % __name__)

def tearDownModule():
    print("tearDownModule %s" % __name__)

class WidgetTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass %s" % __name__)

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass %s" % __name__)

    def setUp(self):
        self.widget = Widget('The widget')

    def tearDown(self):
        self.widget.dispose()

    def test_default_widget_size(self):
        self.assertEqual(self.widget.size(), (50,50),
                         'incorrect default size')

    def test_widget_resize(self):
        self.widget.resize(100,150)
        self.assertEqual(self.widget.size(), (100,150),
                         'wrong size after resize')


def load_tests(loader=None, tests=None, pattern=None):
    if not loader:
        loader = unittest.TestLoader()
    suite_list = [
        loader.loadTestsFromTestCase(WidgetTestCase),
    ]
    return unittest.TestSuite(suite_list)
