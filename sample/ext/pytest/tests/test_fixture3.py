import pytest


@pytest.fixture()
def fixtureFunc():
    print("\n fixture->fixtureFunc")


@pytest.mark.usefixtures("fixtureFunc")
def test_fixture():
    print("in test_fixture")


@pytest.mark.usefixtures("fixtureFunc")
class TestFixture:
    def test_fixture_class(self):
        print("in class with text_fixture_class")


if __name__ == "__main__":
    pytest.main(["-v", "test_fixture.py"])
