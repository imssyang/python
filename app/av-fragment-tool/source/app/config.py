# -*- coding: utf-8 -*-
import yaml
from yaml.loader import SafeLoader


class DataConfig:
    def __init__(self):
        with open("data/main.yml") as file:
            data = yaml.load(file, Loader=SafeLoader)
            self.__dict__ = data
