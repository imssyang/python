import pytest
from selenium import webdriver
import time

@pytest.fixture()
def fixtureFunc():
　　'''实现浏览器的打开和关闭'''
    driver = webdriver.Firefox()
    yield driver
    driver.quit()

def test_search(fixtureFunc):
    '''访问百度首页，搜索pytest字符串是否在页面源码中'''
    driver = fixtureFunc
    driver.get('http://www.baidu.com')
    driver.find_element_by_id('kw').send_keys('pytest')
    driver.find_element_by_id('su').click()
    time.sleep(3)
    source = driver.page_source
    assert 'pytest' in source

if __name__=='__main__':
    pytest.main(['--setup-show', 'test_fixture.py'])
