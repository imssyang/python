# pylint: disable=import-error,import-outside-toplevel
import time
import pytest


@pytest.fixture()
def fixtureFunc():
    from selenium import webdriver

    driver = webdriver.Firefox()
    yield driver
    driver.quit()


def d_test_search(fixtureFunc):
    driver = fixtureFunc
    driver.get("http://www.baidu.com")
    driver.find_element_by_id("kw").send_keys("pytest")
    driver.find_element_by_id("su").click()
    time.sleep(3)
    source = driver.page_source
    assert "pytest" in source


if __name__ == "__main__":
    pytest.main(["--setup-show", "test_fixture.py"])
