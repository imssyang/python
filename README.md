# python3

# 包工具
```
工具：Setuptools (deprecated) -> Distribute (deprecated)    -> pip
模块: distutils               -> distutils && distutils2    -> StandardLibrary
```

# pip工具
```
(源码安装pip)
wget http://pypi.python.org/packages/source/p/pip/pip-0.7.2.tar.gz
tar xzf pip-0.7.2.tar.gz
cd pip-0.7.2
python setup.py install

(命令更新pip)
python -m pip install --upgrade pip                                                                                                                  ✘ 1 master ✱ ◼
  Collecting pip
    Downloading pip-20.3.3-py2.py3-none-any.whl (1.5 MB)
       |████████████████████████████████| 1.5 MB 539 kB/s 
  Installing collected packages: pip
    Attempting uninstall: pip
      Found existing installation: pip 20.1.1
      Uninstalling pip-20.1.1:
        Successfully uninstalled pip-20.1.1
  Successfully installed pip-20.3.3

(pip安装模块)
pip install Markdown               安装
pip install 'Markdown<2.0'         安装指定版本
pip install 'Markdown>2.0,<2.0.3'  安装指定版本
pip install -U Markdown            更新
pip uninstall Markdown             卸载
python -c "import markdown; print markdown.markdown('**Excellent**')"  使用包
pip install path/to/mypackage.tgz                    从本地归档安装（解压后的目录下必须有setup.py）
pip install http://dist.repoze.org/PIL-1.1.6.tar.gz  从网络归档安装（解压后的目录下必须有setup.py）
pip install -e svn+http://svn.colorstudy.com/INITools/trunk#egg=initools-dev 从SVN地址安装（地址结束格式为#egg=packagename）
pip install MyApp -f http://www.example.com/my-packages/  增加搜索地址（存在http://www.example.com/my-packages/MyApp-1.0.tgz时）
```

# CA根证书集合
```
update-ca-certificates --fresh                              更新系统CA目录
export SSL_CERT_DIR=/opt/openssl/ssl/certs:/etc/ssl/certs   Openssl寻找CA目录
```

# 安装certifi
```
当使用Mozilla维护的根证书集合时安装此模块
pip install certifi
pip install git+https://github.com/certifi/python-certifi  从指定地址安装
```

# 寻找包的位置
```
import sys
from pprint import pprint
pprint(sys.path)  打印寻找的目录列表
  /opt/python3/setup
  /opt/python3/lib/python38.zip
  /opt/python3/lib/python3.8
  /opt/python3/lib/python3.8/lib-dynload
  /opt/python3/lib/python3.8/site-packages
  /opt/python3/lib/python3.8/site-packages/six-1.15.0-py3.8.egg
```
# 扩展sys.path
 - 在sys.path中的/opt/python3/lib/python3.8/site-packages目录下增加一个xxx.pth文件(默认存在easy-install.pth), 每行包含一个要增加到sys.path的路径。
   使用相对路径时，路径是相对于.pth文件所在路径的。
 - 设置PYTHONHOME=/opt/python3，一般是python安装目录的前缀，设置后此目录增加到sys.path。
 - 设置PYTHONPATH=/www/python:/opt/py，这两个目录也会增加到sys.path。

# 安装包到../site-packages目录

 - 直接把包放到sys.path中的/opt/python3/lib/python3.8/site-packages目录下。
 - python setup.py install（源码安装）
 - pip install xxx (从pypi下载并安装)

# 上传到pypi.org
```
(使用distutils/setuptools工具)
Create a file $HOME/.pypirc:
  [distutils]
  index-servers =
      pypi

  [pypi]
  username: <username>
  password: <password>

Create a file ~/.pypirc (added pytest optionally):
  [distutils]
  index-servers =
    pypi
    pypitest

  [pypi]
  repository=https://pypi.python.org/pypi
  username=your_username
  password=your_password

  [pypitest]
  repository=https://testpypi.python.org/pypi
  username=your_username
  password=your_password

python setup.py sdist upload  从源码构建出tgz包后上传到pypi

(使用twine工具)
pip install twine
python setup.py clean sdist
TWINE_USERNAME=me TWINE_PASSWORD=passwd twine upload dist/*
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

# virtualenv
https://pypi.org/project/virtualenv/

```
pip install virtualenv            安装
virtualenv pip_test_env           创建pip_test_env环境(创建$HOME/.local/share/virtualenv和pip_test_env目录)
source pip_test_env/bin/activate  激活环境
deactivate                        恢复环境
```
# virtualenvwrapper (依赖virtualenv)
https://pypi.org/project/virtualenvwrapper/
2020-12-26最高在python3.6上测试过

# Pipenv: Python Development Workflow for Humans
整合了pip和virtualenv的功能。


# 示例
[Welcome to The Hitchhiker’s Guide to Packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html)

