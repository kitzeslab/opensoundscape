import inspect
import copy
from pathlib import Path


class PreprocessingError(Exception):
    """Custom exception indicating that a Preprocessor pipeline failed"""

    pass


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


def _run_pipeline(
    pipeline,
    label_df_row,
    break_on_type=None,
    break_on_key=None,
    perform_augmentations=True,
    clip_times=None,
):
    """run the pipeline (until a break point, if specified)

    optionally, can pass a dataframe row specifying the clip times (columns 'start_time' and 'end_time')
    """
    x = Path(label_df_row.name)  # the index contains a path to a file

    # a list of additional things that an action may request from the preprocessor
    sample_info = {
        "_path": Path(label_df_row.name),
        "_labels": copy.deepcopy(label_df_row),
        "_start_time": None if clip_times is None else clip_times["start_time"],
        "_end_time": None if clip_times is None else clip_times["end_time"],
        "_pipeline": pipeline,
    }

    for k, action in pipeline.items():
        if type(action) == break_on_type or k == break_on_key:
            break
        if action.is_augmentation and not perform_augmentations:
            continue
        extra_args = {key: sample_info[key] for key in action.extra_args}
        if action.returns_labels:
            x, labels = action.go(x, **extra_args)
            sample_info["_labels"] = labels
        else:
            x = action.go(x, **extra_args)

    return x, sample_info
