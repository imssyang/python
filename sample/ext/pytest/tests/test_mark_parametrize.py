import pytest

# 单参数单值
@pytest.mark.parametrize("user", ["18221124104", "abc"])
def test(user):
    print(user)
    assert user == "18221124104"
