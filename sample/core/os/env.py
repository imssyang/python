import os


def set_env(k, v):
    os.environ[k] = v


def get_env(k):
    return os.getenv(k)


def show_envs():
    for k, v in sorted(os.environ.items()):
        print(f"{k}: {v}")


show_envs()
