"""utilities for reading and writing action and preprocessor objects to disk"""

import json
import numpy as np
import yaml


# Custom representer for NumPy dtypes
def numpy_dtype_representer(dumper, data):
    return dumper.represent_scalar("!numpy_dtype", data.__name__)


# Custom constructor for NumPy dtypes
def numpy_dtype_constructor(loader, node):
    value = loader.construct_scalar(node)
    return getattr(np, value)


# Create custom loader and dumper classes
class CustomYamlLoader(yaml.Loader):
    pass


class CustomYamlDumper(yaml.Dumper):
    pass


# Register the custom representer and constructor with the custom classes
CustomYamlDumper.add_representer(np.dtype, numpy_dtype_representer)
CustomYamlLoader.add_constructor("!numpy_dtype", numpy_dtype_constructor)


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
