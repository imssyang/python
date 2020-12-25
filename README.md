# python3

# 更新pip版本
```
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
```

# 设置CA证书目录
```
update-ca-certificates --fresh                              更新系统CA目录
export SSL_CERT_DIR=/opt/openssl/ssl/certs:/etc/ssl/certs   Openssl寻找CA目录
```

# 安装certifi(使用Mozilla维护的根证书集合)
pip install certifi
pip install git+https://github.com/certifi/python-certifi

# 示例
[Welcome to The Hitchhiker’s Guide to Packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html)

