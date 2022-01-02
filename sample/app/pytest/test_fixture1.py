# fixture的名字直接作为测试用例的参数
import pytest

@pytest.fixture()
def fixtureFunc():
    print(f"init")
    return 'fixtureFunc'

def test_fixture(fixtureFunc):
    print(f"start")
    print(f"resp: {fixtureFunc}")
    print(f"end")

if __name__=='__main__':
    pytest.main(['-v', 'test_fixture.py'])
