import pytest


@pytest.fixture(autouse=True)
def fixtureFunc():
    print("\n fixture->fixtureFunc")


def test_fixture():
    print("in test_fixture")


class TestFixture(object):
    def test_fixture_class(self):
        print("in class with text_fixture_class")


if __name__ == "__main__":
    pytest.main(["-v", "test_fixture.py"])
