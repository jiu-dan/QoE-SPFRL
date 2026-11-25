import importlib.util
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    module_name = f"{__name__}.{osp.splitext(osp.basename(name))[0]}"
    spec = importlib.util.spec_from_file_location(module_name, pathname)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scenario module from {pathname}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


