import os

print(os.path.splitdrive("upos://x/aaa.mov"))  # ('', 'upos://x/aaa.mov')
print(os.path.split("upos://x/aaa.mov"))  # ('upos://x', 'aaa.mov')

print(os.path.normpath("upos://x/aaa.mov"))  # upos:/x/aaa.mov
print("upos:/x/aaa.mov".split("/"))  # ['upos:', 'x', 'aaa.mov']
