# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor, as_completed


class Test:
    @classmethod
    def run(cls, urls):
        paths = []
        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            workers = [
                executor.submit(cls.transform, urls[index])
                for index, _ in enumerate(urls)
            ]
            for fu in as_completed(workers):
                paths.append(fu.result())
        return paths

    @classmethod
    def transform(cls, url):
        return f"test-{url}"


paths = Test.run(["upos://sucaiboss/aaa.mov", "upos://sucaiboss/bbb.mov"])
print(paths)
