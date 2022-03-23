class PreprocessingError(Exception):
    """Custom exception indicating that a Preprocessor pipeline failed"""

    pass


import inspect


def get_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items()}


def get_reqd_args(func):
    signature = inspect.signature(func)
    return [
        k
        for k, v in signature.parameters.items()
        if v.default is inspect.Parameter.empty
    ]
