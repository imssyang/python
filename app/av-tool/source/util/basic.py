# -*- coding: utf-8 -*-
import os


KB = lambda b: b << 10
MB = lambda b: b << 20
GB = lambda b: b << 30


def version_from_readme(name: str) -> str:
    util_dir = os.path.dirname(os.path.realpath(__file__))
    app_dir = os.path.dirname(os.path.dirname(util_dir))
    readme_path = os.path.join(app_dir, "README.md")
    with open(readme_path, "r") as f:
        t = f.readline().strip()
        version = t[t.rfind('v'):]
        return version


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


