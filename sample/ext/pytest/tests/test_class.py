# Group multiple tests in a class
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")

    def setup_method(self, method):
        if isinstance(method, self.test_one):
            print(f"setup <{method}>")


    def teardown_method(self, method):
        print(f"teardown_ {method}")
