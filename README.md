[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# python3

# 包工具

```txt
工具：Setuptools (deprecated) -> Distribute (deprecated)    -> pip
模块: distutils               -> distutils && distutils2    -> StandardLibrary
```

# pip工具

```txt
(源码安装pip)
wget http://pypi.python.org/packages/source/p/pip/pip-0.7.2.tar.gz
tar xzf pip-0.7.2.tar.gz
cd pip-0.7.2
python setup.py install

(命令更新pip)
python -m pip install --upgrade pip
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
pip install pysocks                支持socks代理
pip install -r requirements.txt --proxy='socks5://127.0.0.1:1080'
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

# pipx
```txt
pip install --user pipx     安装到$HOME/.local
pip install pipx            安装
pip install -U pipx         更新
pipx ensurepath             将安装位置加入到$PATH
```

# CA根证书集合
```txt
update-ca-certificates --fresh                              更新系统CA目录
export SSL_CERT_DIR=/opt/openssl/ssl/certs:/etc/ssl/certs   Openssl寻找CA目录
```

# 安装certifi

```shell
当使用Mozilla维护的根证书集合时安装此模块
pip install certifi
pip install git+https://github.com/certifi/python-certifi  从指定地址安装
```

# 寻找包的位置

```shell
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

```shell
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

```shell
pip install virtualenv            安装
virtualenv pip_test_env           创建pip_test_env环境(创建$HOME/.local/share/virtualenv和pip_test_env目录)
source pip_test_env/bin/activate  激活环境
deactivate                        恢复环境
```

# virtualenvwrapper (依赖virtualenv)

https://pypi.org/project/virtualenvwrapper/
2020-12-26最高在python3.6上测试过

```shell
pip install virtualenvwrapper         安装
export WORKON_HOME=/opt/python3/envs  设置工作目录
mkdir -p $WORKON_HOME                 创建工作目录
source /opt/python3/bin/virtualenvwrapper.sh  安装函数
mkvirtualenv env1                     创建环境
rmvirtualenv env1                     删除环境
workon env1                           激活环境
workon                                列出环境
```

# Pipenv: Python Development Workflow for Humans
整合了pip和virtualenv的功能。
https://github.com/pypa/pipenv

# Egg包

使用egg包的两个方式：
 - *.egg拷贝到../site-packages后，在../site-packages/easy-install.pth文件中增加一行：./xxxx.egg.
 - 运行时将egg文件添加到环境变量PYTHONPATH。例如PYTHONPATH=xxx.egg python xxx.py.

# Wheel包 (新标准)

```shell
pip install wheel            安装
vim setup.cfg                创建此文件配置为wheel格式（可选）
  [bdist_wheel]
  universal = 1              (仅在考虑兼容python2时)
python setup.py bdist_wheel  编译后打包成wheel格式
```

# pylint

```shell
pip install pylint
pip install pylint-gitlab
```

# pre-commit

[Supported hooks](https://pre-commit.com/hooks.html)

```shell
pip install pre-commit
PROJECT/.pre-commit-config.yaml     配置安装的钩子类型
pre-commit install --install-hooks  安装hook并打印安装位置
pre-commit uninstall                卸载hook
pre-commit run --all-files          在所有文件上运行hook
pre-commit run --files              在指定文件上运行hook
pre-commit run --from-ref origin/HEAD --to-ref HEAD   只检查已经更改的文件
git config --global init.templateDir ~/.git-template
pre-commit init-templatedir ~/.git-template           新clone仓库自动配置pre-commit
git commit --no-verify              提交时禁用pre-commit
~/.cache/pre-commit                 默认CACHE位置
PRE_COMMIT_HOME                     环境变量，可代替默认Cache位置
XDG_CACHE_HOME                      环境变量，可代替PRE_COMMIT_HOME位置
```

# commit-msg

# PyTorch

```shell
# PyTorch v1.11.0 & CUDA 11.3
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# PyTorch v1.11.0 & CPU
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

# PyQt6

https://www.riverbankcomputing.com/

# http

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory /tmp/
```

# Jupyterlab

```bash
$ python -c "from jupyter_server.auth import passwd; print(passwd('XXX'))"

$ vi jupyter_lab_config.py
c.ServerApp.password = u'argon2:$argon2id$v=19$m=10240,t=10,p=8$nfU35Ct7vbjCiWxPhxLCMg$T1TSPT23ZvKfi7ykHioUSz/rbkvkTVClFWu5F7mZROQ'

$ jupyter lab build --dev-build=False --minimize=False
$ jupyter labextension uninstall @jupyterlab/celltags-extension
```
[Common Directories and File Locations](https://docs.jupyter.org/en/latest/use/jupyter-directories.html)

# uwsgi

```bash
pip3 install uwsgi     (maybe)
pip3 install uwsgitop

uwsgi ... --stats :5001
uwsgitop localhost:5001
```

# 示例
[Welcome to The Hitchhiker’s Guide to Packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html)
[Awesome Python](https://python.libhunt.com/)
