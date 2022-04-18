import inspect
import copy
from pathlib import Path
import pandas as pd


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
    augmentation_on=True,
    clip_times=None,
    sample_duration=None,
):
    """run the pipeline (until a break point, if specified)

    optionally, can pass a clip_times Series specifying 'start_time' 'end_time'
    """
    x = Path(label_df_row.name)  # the index contains a path to a file

    # a list of additional things that an action may request from the preprocessor
    sample_info = {
        "_path": Path(label_df_row.name),
        "_labels": copy.deepcopy(label_df_row),
        "_start_time": None if clip_times is None else clip_times["start_time"],
        "_sample_duration": sample_duration,
        "_pipeline": pipeline,
    }

    for k, action in pipeline.items():
        if type(action) == break_on_type or k == break_on_key:
            break
        if action.is_augmentation and not augmentation_on:
            continue
        extra_args = {key: sample_info[key] for key in action.extra_args}
        if action.returns_labels:
            x, labels = action.go(x, **extra_args)
            sample_info["_labels"] = labels
        else:
            x = action.go(x, **extra_args)

    return x, sample_info


def insert_before(series, idx, name, value):
    """insert an item before a spcific index in a series"""
    i = list(x.index).index(idx)
    part1 = x[0:i]
    part2 = x[i:]
    return part1.append(pd.Series([value], index=[name])).append(part2)


def insert_after(series, idx, name, value):
    """insert an item after a spcific index in a series"""
    i = list(series.index).index(idx)
    part1 = series[0 : i + 1]
    part2 = series[i + 1 :]
    return part1.append(pd.Series([value], index=[name])).append(part2)
