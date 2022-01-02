
# PoorTest: Having each test share the same class instance would be very detrimental.
class TestClassDemoInstance:
    def test_one(self):
        assert 0

    def test_two(self):
        assert 0
