# pylint: disable=consider-using-f-string
import pytest


@pytest.fixture()
def fixtureFunc():
    return "fixtureFunc"


def test_fixture(fixtureFunc):
    print("call func {}".format(fixtureFunc))


class TestFixture:
    def test_fixture_class(self, fixtureFunc):
        print('call class fixture "{}"'.format(fixtureFunc))


if __name__ == "__main__":
    pytest.main(["-v", "test_fixture.py"])
