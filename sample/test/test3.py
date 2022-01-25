import os

os.environ['VXCODE_DRY_RUN'] = "1"

print(os.getenv('VXCODE_POOL_SIZE'))
print(int(os.getenv('VXCODE_DRY_RUN')))