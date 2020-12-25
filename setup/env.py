import os

# print environment variables
#print(os.environ)

# print the environment variables in a better readable way
for k, v in os.environ.items():
    print(f'{k}={v}')

