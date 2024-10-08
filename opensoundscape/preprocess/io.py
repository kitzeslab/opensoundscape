"""utilities for serializing, reading, and writing Action and Preprocessor objects to/from files and dictionaries"""

import json
import numpy as np
import yaml
from types import MethodType, FunctionType


def build_name(method_or_object):
    """return the full function or class name

    Args:
        method_or_object: a method, object, or class

    Returns: a string like "opensoundscape.module.Class.method"
        - Note: if defined in a script, looks like "__main__.my_function"

    """
    prefix = method_or_object.__module__
    if isinstance(method_or_object, (MethodType, FunctionType, type)):
        # its a method/function or a class
        return f"{prefix}.{method_or_object.__qualname__}"
    return f"{prefix}.{type(method_or_object).__qualname__}"


# Custom representer for NumPy dtypes
def numpy_dtype_representer(dumper, data):
    return dumper.represent_scalar("!numpy_dtype", data.__name__)


# Custom constructor for NumPy dtypes
def numpy_dtype_constructor(loader, node):
    value = loader.construct_scalar(node)
    return getattr(np, value)


# Create custom loader and dumper classes
class CustomYamlLoader(yaml.Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_constructor("!numpy_dtype", numpy_dtype_constructor)


class CustomYamlDumper(yaml.Dumper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_representer(np.dtype, numpy_dtype_representer)


class NumpyTypeEncoder(json.JSONEncoder):
    """replace numpy dtypes with strings & prefix `numpy_dtype_`

    otherwise, can't serialize numpy dtypes as the value in a dictionary"""

    def default(self, obj):
        if isinstance(obj, type) and issubclass(obj, np.generic):
            return f"numpy_dtype_{obj.__name__}"
        return super().default(obj)


class NumpyTypeDecoder(json.JSONDecoder):
    """recursively modify dictionary to change "numpy_dtype_..." strings to numpy dtypes

    See also: NumpyTypeEncoder"""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj, dict):
            return {key: self._convert_dtype(value) for key, value in obj.items()}
        return obj

    def _convert_dtype(self, value):
        if isinstance(value, str) and value.startswith("numpy_dtype_"):
            return getattr(np, value.replace("numpy_dtype_", ""))
        elif isinstance(value, dict):
            return {key: self._convert_dtype(val) for key, val in value.items()}
        else:
            return value
