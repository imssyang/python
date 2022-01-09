import sys
import importlib.util
def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

lazy_typing = lazy_import("typing")
print(lazy_typing)                # a real module object
print(lazy_typing.TYPE_CHECKING)  # but it is not loaded in memory yet