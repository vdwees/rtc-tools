import inspect
import os


def local_function():
    pass


def data_path():
    return os.path.join(os.path.dirname(os.path.abspath(inspect.getsourcefile(local_function))), 'data')
