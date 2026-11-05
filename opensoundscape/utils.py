"""Utilities for opensoundscape"""

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from matplotlib.colors import LinearSegmentedColormap
import torch
import random
import datetime
import pytz
import opensoundscape
from aru_metadata_parser.parse import parse_aru_file_start_time


class GetDurationError(ValueError):
    """raised if librosa.get_duration(path=f) causes an error"""

    pass


def isNan(x):
    """check for nan by equating x to itself"""
    return not x == x


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def identity(x):
    """return the input unchanged"""
    return x


def overlap(r1, r2):
    """ "calculate the amount of overlap between two real-numbered ranges

    ranges must be [low,high] where low <= high"""
    assert r1[1] >= r1[0]
    assert r2[1] >= r2[0]
    lower_bound = max(r1[0], r2[0])
    upper_bound = min(r1[1], r2[1])
    return max(0, upper_bound - lower_bound)


def overlap_fraction(r1, r2):
    """ "calculate the fraction of r1 (low, high range) that overlaps with r2"""
    return overlap(r1, r2) / (r1[1] - r1[0])


def inrange(x, r):
    """return true if x is in range [r[0],r1] (inclusive)"""
    return x >= r[0] and x <= r[1]


def binarize(x, threshold):
    """return a list of 0, 1 by thresholding vector x"""
    if len(np.shape(x)) > 2:
        raise ValueError("shape must be 1 dimensional or 2 dimensional")

    if len(np.shape(x)) == 2:
        return [[1 if xi > threshold else 0 for xi in row] for row in x]

    return [1 if xi > threshold else 0 for xi in x]


def rescale_features(X, rescaling_vector=None):
    """rescale all features by dividing by the max value for each feature

    optionally provide the rescaling vector (1xlen(X) np.array),
    so that you can rescale a new dataset consistently with an old one

    returns rescaled feature set and rescaling vector"""
    if rescaling_vector is None:
        rescaling_vector = 1 / np.nanmax(X, 0)
    rescaled_x = np.multiply(X, rescaling_vector).tolist()
    return rescaled_x, rescaling_vector


def min_max_scale(array, feature_range=(0, 1)):
    """rescale vaues in an a array linearly to feature_range"""
    bottom, top = feature_range
    array_min = np.min(array)
    array_max = np.max(array)
    scale_factor = (top - bottom) / (array_max - array_min)
    return scale_factor * (array - array_min) + bottom


def linear_scale(array, in_range=(0, 1), out_range=(0, 255)):
    """Translate from range in_range to out_range

    Inputs:
        in_range: The starting range [default: (0, 1)]
        out_range: The output range [default: (0, 255)]

    Outputs:
        new_array: A translated array
    """
    scale_factor = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
    return scale_factor * (array - in_range[0]) + out_range[0]


def jitter(x, width, distribution="gaussian"):
    """
    Jitter (add random noise to) each value of x

    Args:
        x: scalar, array, or nd-array of numeric type
        width: multiplier for random variable (stdev for 'gaussian' or r for 'uniform')
        distribution: 'gaussian' (default) or 'uniform'
            if 'gaussian': draw jitter from gaussian with mu = 0, std = width
            if 'uniform': draw jitter from uniform on [-width, width]
    Returns:
        jittered_x: x + random jitter
    """
    if distribution == "gaussian":
        return np.array(x) + np.random.normal(0, width, size=np.shape(x))
    elif distribution == "uniform":
        return np.array(x) + np.random.uniform(-1 * width, width, size=np.shape(x))
    raise ValueError(
        f"distribution must be 'gaussian' or 'uniform'. Got {distribution}."
    )


def generate_clip_times_df(
    full_duration,
    clip_duration,
    clip_overlap=None,
    overlap_fraction=None,
    clip_step=None,
    final_clip="extend",
    rounding_precision=10,
):
    """generate start and end times for even-lengthed clips

    The behavior for incomplete final clips at the end of the full_duration
    depends on the final_clip parameter.

    This function only creates a dataframe with start and end times, it does
    not perform any actual trimming of audio or other objects.

    Args:
        full_duration: The amount of time (seconds) to split into clips
        clip_duration (float):  The duration in seconds of the clips
        clip_overlap (float):   The overlap of the clips in seconds
        overlap_fraction (float): The overlap of the clips as a fraction of clip_duration
        clip_step (float):      The increment in seconds between starts of consecutive clips
            - must only specify one of clip_overlap, overlap_fraction, or clip_step
            - if all are None, overlap is set to 0
        final_clip (str):       Behavior if final_clip is less than clip_duration
            seconds long. By default, discards remaining time if less than
            clip_duration seconds long [default: None].
            Options:
                - None:         Discard the remainder (do not make a clip)
                - "extend":     Extend the final clip beyond full_duration with zeros to reach
                  clip_duration length
                - "remainder":  Use only remainder of full_duration (final clip will be shorter than
                  clip_duration)
                - "full":       Increase overlap with previous clip to yield a clip with
                  clip_duration length. Note: returns entire original audio if it is shorter than
                  clip_duration
        rounding_precision (int or None): number of decimals to round start/end times to
            - pass None to skip rounding

    Returns:
        clip_df: DataFrame with columns for 'start_time' and 'end_time' of each clip
    """
    if not final_clip in ["remainder", "full", "extend", None]:
        raise ValueError(
            f"final_clip must be 'remainder', 'full', 'extend',"
            f"or None. Got {final_clip}."
        )

    overspecified_overlap_err = (
        "only one of clip_overlap, overlap_fraction, or clip_step can be specified"
    )
    if clip_overlap is not None:
        if overlap_fraction is not None or clip_step is not None:
            raise ValueError(overspecified_overlap_err)
        assert (
            clip_overlap < clip_duration
        ), "clip_overlap must be less than clip_duration"
    elif overlap_fraction is not None:
        if clip_overlap is not None or clip_step is not None:
            raise ValueError(overspecified_overlap_err)
        assert 0 <= overlap_fraction < 1, "overlap_fraction must be in [0, 1)"
        clip_overlap = overlap_fraction * clip_duration
    elif clip_step is not None:
        # allow values outside of [0, clip_duration]
        if clip_overlap is not None or overlap_fraction is not None:
            raise ValueError(overspecified_overlap_err)
        clip_overlap = clip_duration - clip_step
    else:
        clip_overlap = 0

    # Lists of start and end times for clips
    increment = clip_duration - clip_overlap
    starts = np.arange(0, full_duration, increment)
    if rounding_precision is not None:
        starts = starts.round(rounding_precision)
    ends = starts + clip_duration

    # Handle the final_clip
    if final_clip is None:
        # Throw away any clips with end times beyond full_duration
        keeps = ends <= full_duration
        ends = ends[keeps]
        starts = starts[keeps]
    elif final_clip == "remainder":
        # Trim clips with end times beyond full_duration to full_duration
        ends[ends > full_duration] = full_duration
    elif final_clip == "full":
        # Increase the overlap of any clips with end_time past full_duration
        # so that they end at full_duration
        # can result in duplicates of the same final_clip - removed while returning
        clip_idxs_to_shift = ends > full_duration
        starts[clip_idxs_to_shift] -= ends[clip_idxs_to_shift] - full_duration
        ends[clip_idxs_to_shift] = full_duration

        # set the start timestamp to 0 for shorter clips
        # to avoid negative start times and return original clip
        starts[starts < 0] = 0

    elif final_clip == "extend":
        # Keep the end values that extend beyond full_duration
        pass

    return pd.DataFrame({"start_time": starts, "end_time": ends}).drop_duplicates()


def cast_np_to_native(x):
    """if the input is a numpy integer or floating type, cast to native Python int or float

    otherwise, input is unaffected
    """
    # timedelta doesn't like np types, fix issue #928
    if isinstance(x, np.integer):
        return int(x)
    elif isinstance(x, np.floating):
        return float(x)
    return x


def make_clip_df(
    files,
    clip_duration,
    clip_overlap=None,
    overlap_fraction=None,
    clip_step=None,
    final_clip="extend",
    return_invalid_samples=False,
    raise_exceptions=False,
    audio_root=None,
):
    """generate df of fixed-length clip start/end times for a set of files

    Used internally to prepare a dataframe listing clips of longer audio files

    This function creates a single dataframe with audio files as
    the index and columns: 'start_time', 'end_time'. It will list
    clips of a fixed duration from the beginning to end of each audio file.

    Note: if a label dataframe is passed as `files`, the labels for each file
    will be copied to all clips having the corresponding file. If the label dataframe
    contains multiple rows for a single file, the labels in the _first_ row containing
    the file path are used as labels for resulting clips.

    Args:
        files: list of audio file paths, or dataframe with file path as index
            - if dataframe, columns represent classes and values represent
            class labels. Labels for a file will be copied to all clips
            belonging to that file in the returned clip dataframe.
        clip_duration (float): see generate_clip_times_df
        clip_overlap (float): see generate_clip_times_df
        overlap_fraction (float): see generate_clip_times_df
        clip_step (float): see generate_clip_times_df
        final_clip (str): see generate_clip_times_df
        return_invalid_samples (bool): if True, returns additional value,
            a list of samples that caused exceptions
        raise_exceptions (bool): if True, if exceptions are raised when attempting
            to check the duration of an audio file, the exception will be raised.
            If False [default], adds a row to the dataframe with np.nan for
            'start_time' and 'end_time' for that file path.
        audio_root: optionally pass a root directory (pathlib.Path or str)
            - `audio_root` is prepended to each file path
            - if None (default), `files` must contain full paths to files

    Returns:
        clip_df: dataframe multi-index ('file','start_time','end_time')
            - if files is a dataframe, will contain same columns as files
            - otherwise, will have no columns

        if return_invalid_samples==True, returns (clip_df, invalid_samples)

    Note: default behavior for raise_exceptions is the following:
        if an exception is raised (for instance, trying to get the duration of the file),
        the dataframe will have one row with np.nan for 'start_time' and 'end_time' for that
        file path.
    """

    label_df = None  # assume no labels to begin with, just a list of paths
    if isinstance(files, pd.DataFrame):
        file_list = files.index.values
        # use the dataframe as labels, keeping each column as a class
        # if paths are duplicated in index, keep only the first of each
        label_df = files[~files.index.duplicated(keep="first")]
    elif isinstance(files, (str, Path)):
        file_list = [files]  # be lenient, turn single path into list
    else:
        assert hasattr(files, "__iter__"), (
            f"`files` should be a dataframe with paths as "
            f"the index, or an iterable of file paths. Got {type(files)}."
        )
        file_list = files

    clip_dfs = []
    invalid_samples = set()
    idx_cols = ["file", "start_time", "end_time"]
    for f in file_list:
        if audio_root is None:
            path = f
        else:
            path = Path(audio_root) / Path(f)
        try:
            t = librosa.get_duration(path=path)
            clips = generate_clip_times_df(
                full_duration=t,
                clip_duration=clip_duration,
                clip_overlap=clip_overlap,
                overlap_fraction=overlap_fraction,
                clip_step=clip_step,
                final_clip=final_clip,
            )
            clips["file"] = f

        except Exception as exc:
            if raise_exceptions:
                raise GetDurationError(f"Exception on file {f}") from exc
            else:
                # make one row for this file with nan for start/end times
                clips = pd.DataFrame(
                    {"file": [f], "start_time": np.nan, "end_time": np.nan}
                )
                invalid_samples.add(f)

        if label_df is not None:
            # copy labels for this file to all of its clips
            clips[label_df.columns] = label_df.loc[f].values

        clip_dfs.append(clips)

    if len(clip_dfs) > 0:
        clip_df = pd.concat(clip_dfs).set_index(idx_cols)
    else:
        # warnings.warn(
        #     f"No clips were created from file_list of length {len(file_list)}"
        # )
        # create an empty dataframe with the expected index and columns
        label_cols = [] if label_df is None else label_df.columns
        clip_df = pd.DataFrame(columns=idx_cols + label_cols).set_index(idx_cols)

    if return_invalid_samples:
        return clip_df, invalid_samples
    else:
        return clip_df


def generate_opacity_colormaps(
    colors=["#067bc2", "#43a43d", "#ecc30b", "#f37748", "#d56062"]
):
    """Create a colormap for each color from transparent to opaque"""
    colormaps = []

    for color in colors:
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", [(0, 0, 0, 0), color], N=256
        )
        colormaps.append(cmap)

    return colormaps


def set_seed(seed, verbose=False):
    """Set random state across different libraries for reproducibility

    Args:
        seed (int): Number to fix random number generators to a specific start.
        verbose (bool, optional): Print set seed. Defaults to True.
    """
    if verbose:
        print(f"Random state set with seed {seed}")

    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# utilities for guessing deployment name from file path


def parent_folder_name(file_path):
    """Utility function to extract the parent folder name from a file path"""
    return Path(file_path).parent.name


def two_parents_name(file_path):
    """Utility function to extract "grandparent_parent" folder name from a file path"""
    p = Path(file_path).parent
    gp = p.parent
    return f"{gp.name}_{p.name}"


def second_parent_name(file_path):
    """Utility function to extract the second parent folder name from a file path"""
    return Path(file_path).parent.parent.name


def filename_first_part(file_path):
    """Utility function to extract the part of the filename before the first underscore from a file path"""
    return Path(file_path).stem.split("_")[0]


class GetRecordingStartTimeError(ValueError):
    """Raised when a recording start time cannot be resolved in strict mode."""


def _read_recording_start_time(file, parse_timestamp_from_filename):
    """Read a datetime from metadata, optionally falling back to the filename."""
    metadata = opensoundscape.audio.parse_metadata(file) or {}
    recording_start_time = metadata.get("recording_start_time")
    if isinstance(recording_start_time, datetime.datetime):
        return recording_start_time
    if parse_timestamp_from_filename:
        return parse_aru_file_start_time(file)
    raise ValueError("metadata does not contain a recording start timestamp")


def _localize_recording_start_time(recording_start_time, file_timezone):
    """Return an aware datetime, validating any supplied recorder timezone."""
    if recording_start_time.utcoffset() is None:
        if file_timezone is None:
            raise ValueError("provide file_timezone for naive timestamps")
        # Raise if daylight saving time makes the timestamp ambiguous or nonexistent
        return file_timezone.localize(recording_start_time, is_dst=None)

    if file_timezone is not None:
        converted_start_time = recording_start_time.astimezone(file_timezone)
        if converted_start_time.utcoffset() != recording_start_time.utcoffset():
            raise ValueError("file_timezone conflicts with timestamp offset")
    return recording_start_time


def _select_recordings(
    files,
    predicate,
    target_timezone,
    parse_timestamp_from_filename,
    file_timezone,
    raise_exceptions,
):
    """Resolve timestamps, enact exception policy, and filter using predicate function.

    Args:
        files: Iterable of path-like objects; order and duplicates are preserved.
        predicate: Function passed the datetime objects resolved from a file,
        returns a boolean indicating if the file should be included in the filtered
        results
        target_timezone: pytz timezone name in which calendar dates are compared.
        parse_timestamp_from_filename: Attempt aru-metadata-parser filename
        parsing only if metadata lacks a usable timestamp.
        file_timezone: pytz timezone name for naive timestamps. For aware timestamps,
        its offset must agree with the encoded offset at that instant.
        raise_exceptions: If False, skip files with timestamp errors. If True, raise
        GetRecordingStartTimeError with the original cause. Includes missing
        timestamps, DST failures, and timezone conflicts.

    Returns:
        List of original path objects whose datetimes match the predicate function.

    Raises:
        ValueError: Invalid bounds, timezone names, or flags.
        GetRecordingStartTimeError: File timestamp failure when raise_exceptions=True.
    """
    target_timezone = pytz.timezone(target_timezone)
    if file_timezone is not None:
        file_timezone = pytz.timezone(file_timezone)

    selected = []
    for file in files:
        try:
            recording_start_time = _read_recording_start_time(
                file,
                parse_timestamp_from_filename=parse_timestamp_from_filename,
            )
            recording_start_time = _localize_recording_start_time(
                recording_start_time,
                file_timezone=file_timezone,
            )
            recording_start_time = recording_start_time.astimezone(target_timezone)
        except Exception as exc:
            if raise_exceptions:
                raise GetRecordingStartTimeError(f"Exception on file {file}") from exc
            continue
        if predicate(recording_start_time):
            selected.append(file)
    return selected


def select_files_in_date_range(
    files,
    first_date,
    last_date,
    target_timezone,
    parse_timestamp_from_filename=False,
    file_timezone=None,
    raise_exceptions=False,
):
    """Selects recordings with start times within a date range, including start and end date

    Args:
        files: Iterable of path-like objects; order and duplicates are preserved.
        first_date: Inclusive datetime.date bound (not datetime.datetime).
        last_date: Inclusive datetime.date bound, at least first_date.
        target_timezone: pytz timezone name in which calendar dates are compared.
        parse_timestamp_from_filename: Attempt aru-metadata-parser filename parsing
        only if metadata lacks a usable timestamp. Defaults to False.
        file_timezone: pytz timezone name for naive timestamps. For aware timestamps,
        its offset must agree with the encoded offset at that instant.
        raise_exceptions: If False (default), skip files with timestamp errors. If
        True, raise GetRecordingStartTimeError with the original cause. Includes
        missing timestamps, DST failures, and timezone conflicts.

    Returns:
        List of original path objects whose recording starts match the range.

    Raises:
        ValueError: Invalid bounds, timezone names, or flags.
        GetRecordingStartTimeError: File timestamp failure when raise_exceptions=True.

    Examples:
        >>> from datetime import date
        >>> select_files_in_date_range([], date(2024, 1, 1), date(2024, 1, 2), 'UTC')
        []

    Notes:
        Only recording starts are considered, not recording overlap or file
        modification times. Metadata takes precedence over filenames. If timezone is
        not found in metadata or filename, this will require file_timezone.
        Ambiguous/nonexistent daylight saving times are unresolved (will raise an
        exception or skip). Filename recognition, including fuzzy parsing, is best
        effort.
    """
    if any(
        not isinstance(d, datetime.date) or isinstance(d, datetime.datetime)
        for d in (first_date, last_date)
    ):
        raise ValueError("bounds must be datetime.date objects, not datetimes")
    if first_date > last_date:
        raise ValueError("first_date must be <= last_date")

    def predicate(dt):
        return first_date <= dt.date() <= last_date

    return _select_recordings(
        files,
        predicate,
        target_timezone,
        parse_timestamp_from_filename,
        file_timezone,
        raise_exceptions,
    )


def select_files_in_time_range(
    files,
    earliest_time,
    latest_time,
    target_timezone,
    parse_timestamp_from_filename=False,
    file_timezone=None,
    raise_exceptions=False,
):
    """Select recordings by time of day they started, inclusive of end values

    Args:
        files: Iterable of path-like objects.
        earliest_time: Timezone-free datetime.time lower boundary.
        latest_time: Timezone-free datetime.time upper boundary. If earlier than earliest_time, the range wraps midnight. Equal bounds match exactly.
        target_timezone: pytz timezone name for wall-clock comparisons.
        parse_timestamp_from_filename: Enable best-effort filename fallback; metadata
        is preferred. Defaults to False.
        file_timezone: Timezone name for naive timestamps, or an offset consistency
        check for aware timestamps; never overrides an encoded timezone.
        raise_exceptions: If False (default), skip files with timestamp errors. If
        True, raise GetRecordingStartTimeError with the original cause.
        Includes missing timestamps, DST failures, and timezone conflicts.

    Returns:
        List of original paths preserving order and duplicates.

    Raises:
        ValueError: Invalid bounds, timezone names, or flags.
        GetRecordingStartTimeError: File timestamp failure when raise_exceptions=True.

    Notes:
        Only recording starts are considered, not recording overlap or file
        modification times. Metadata takes precedence over filenames. If timezone is
        not found in metadata or filename, this will require file_timezone.
        Ambiguous/nonexistent daylight saving times are unresolved (will raise an
        exception or skip). Filename recognition, including fuzzy parsing, is best
        effort.
    """
    if any(
        not isinstance(t, datetime.time) or t.tzinfo is not None
        for t in (earliest_time, latest_time)
    ):
        raise ValueError("bounds must be timezone-free datetime.time objects")

    def predicate(dt):
        t = dt.time()
        if earliest_time <= latest_time:
            return earliest_time <= t <= latest_time
        return t >= earliest_time or t <= latest_time

    return _select_recordings(
        files,
        predicate,
        target_timezone,
        parse_timestamp_from_filename,
        file_timezone,
        raise_exceptions,
    )
