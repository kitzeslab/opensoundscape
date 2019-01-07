import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from itertools import repeat
from scipy import signal, ndimage
from librosa import load, to_mono
from skimage.morphology import remove_small_objects
from modules.db_utils import init_client
from modules.db_utils import close_client
from modules.db_utils import write_spectrogram
from modules.utils import return_cpu_count


def generate_segments_from_binary_spectrogram(binary_spec, buffer):
    """Identify feature bounding boxes

    Given a binary spectrogram, label the segments and find the bounding
    boxes (plus a buffer) around each feature segment.

    Args:
        binary_spec: A binary spectrogram
        buffer: A number of pixels to add to the bounding boxes

    Returns:
        A dataframe containing the bounding boxes
    """
    # Label the segments and get raw bounding boxes
    labeled_segments, num_of_segments = ndimage.label(binary_spec)
    raw_bboxes = [
        ndimage.find_objects(labeled_segments == x)
        for x in range(1, num_of_segments + 1)
    ]

    def is_too_small(pixel):
        val = pixel - buffer
        return val if val > 0 else 0

    def is_too_large(pixel, axis):
        val = pixel + buffer
        return val if val < binary_spec.shape[axis] else binary_spec.shape[axis]

    data = []
    for box in raw_bboxes:
        y_min = is_too_small(box[0][0].start)
        y_max = is_too_large(box[0][0].stop, 0)
        x_min = is_too_small(box[0][1].start)
        x_max = is_too_large(box[0][1].stop, 1)
        data.append([x_min, x_max, y_min, y_max])

    df = pd.DataFrame(data, columns=["x_min", "x_max", "y_min", "y_max"])
    df.sort_values("x_min", axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def scaled_median_filter(spec, factor):
    """Filter via scaled median threshold

    Given a spectrogram, filter rows and columns by the median with a scaling
    factor anything below the threshold is set to 0.0. _filter_by_scaled_median
    is the function which allows us to do this.  By returning the logical_and
    we are converting the spectrogram to binary for additional processing.

    Args:
        spec: The spectrogram generated from signal.spectrogram
        factor: The factor to pass to the median filter

    Returns:
        A binary spectrogram
    """

    def _filter_by_scaled_median(arr):
        _temp = np.copy(arr)
        median = factor * np.median(_temp)
        for i, val in enumerate(_temp):
            if val < median:
                _temp[i] = 0.0
        return _temp

    row_filt = np.apply_along_axis(_filter_by_scaled_median, 0, spec)
    col_filt = np.apply_along_axis(_filter_by_scaled_median, 1, spec)
    return np.logical_and(row_filt, col_filt)


def frequency_based_spectrogram_filter(spec, freq, low_freq_thr, high_freq_thr):
    """Filter any useless low and high frequency ranges

    Given a spectrogram and frequencies generated from signal.spectrogram
    filter any low and high frequencies we aren't interested in analyzing.

    Args:
        spec: The input spectrogram from signal.spectrogram
        freq: The input frequencies from signal.spectrogram
        low_freq_thr: The low frequency threshold
        high_freq_thr: The high frequency threshold

    Returns:
        A tuple containing the new spectrogram and frequencies
    """
    indices = np.argwhere((freq > low_freq_thr) & (freq < high_freq_thr)).flatten()
    return spec[indices, :], freq[indices]


def chunk_preprocess(chunk, config):
    """Preprocess all images

    This is a super function which provides all of the functionality to
    preprocess many wav file for model fitting.

    Args:
        chunk: A chunk of files to process
        config: The parsed ini file for this particular run

    Returns:
        Nothing

    Raises:
        Nothing
    """

    # Before running the chunk, we need to initialize the global MongoDB client
    init_client(config)

    for label in chunk:
        preprocess(label, config)

    # Don't forget to close the client!
    close_client()

    return


def preprocess(label, config):
    """Preprocess all images

    This is a super function which provides all of the functionality to
    preprocess many wav file for model fitting. Later, this function will call
    an external algorithm to do image processing.

    Args:
        label: The label e.g. train/001.wav
        config: The parsed ini file for this particular run

    Returns:
        if read write to DB defined (`db_rw` in INI)
            Write data to MongoDB w/ `db_name`
        else:
            Return the bounding box dataframe, spectrogram, and normalization
            factor
    Raises:
        Nothing
    """

    # Resample
    samples, sample_rate = load(
        f"{config['general']['data_dir']}/{label}",
        mono=False,  # Don't automatically load as mono, so we can warn if we force to mono
        sr=config["spect_gen"].getfloat("resample_rate"),
        res_type=config["spect_gen"]["resample_type"],
    )

    # Force to mono if wav has multiple channels
    if samples.ndim > 1:
        samples = to_mono(samples)
        print(
            f"WARNING: Multiple-channel file detected ({config['general']['data_dir']}/{label}). Automatically mixed to mono."
        )

    # Generate Spectrogram
    nperseg = config["spect_gen"].getint("spectrogram_segment_length")
    noverlap = int(
        nperseg * config["spect_gen"].getfloat("spectrogram_overlap") / 100.0
    )
    frequencies, times, spectrogram = signal.spectrogram(
        samples,
        sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
    )

    # Frequency Selection
    spectrogram, frequencies = frequency_based_spectrogram_filter(
        spectrogram,
        frequencies,
        config["spect_gen"].getint("low_freq_thresh"),
        config["spect_gen"].getint("high_freq_thresh"),
    )

    # Z-score normalization, need mean and l2 norm later
    spectrogram_mean = spectrogram.mean()
    spectrogram_l2_norm = np.linalg.norm(spectrogram, ord=2)
    spectrogram = (spectrogram - spectrogram_mean) / spectrogram_l2_norm

    # Scaled Median Column/Row Filters
    binary_spectrogram = scaled_median_filter(
        spectrogram, config["spect_gen"].getfloat("median_filter_factor")
    )

    # Binary Closing
    binary_spectrogram = ndimage.morphology.binary_closing(
        binary_spectrogram,
        structure=np.ones(
            (
                config["spect_gen"].getint("binary_closing_kernel_height"),
                config["spect_gen"].getint("binary_closing_kernel_width"),
            )
        ),
    )

    # Binary Dilation
    binary_spectrogram = ndimage.morphology.binary_dilation(
        binary_spectrogram,
        structure=np.ones(
            (
                config["spect_gen"].getint("binary_dilation_kernel_height"),
                config["spect_gen"].getint("binary_dilation_kernel_width"),
            )
        ),
    )

    # Binary Median Filter
    binary_spectrogram = ndimage.median_filter(
        binary_spectrogram,
        size=(
            config["spect_gen"].getint("median_filter_kernel_height"),
            config["spect_gen"].getint("median_filter_kernel_width"),
        ),
    )

    # Remove Small Objects
    binary_spectrogram = remove_small_objects(
        binary_spectrogram, config["spect_gen"].getint("small_objects_kernel_size")
    )

    # Get the bounding box dataframe from the labeled segments
    bboxes_df = generate_segments_from_binary_spectrogram(
        binary_spectrogram, config["spect_gen"].getint("segment_pixel_buffer")
    )

    # Write to DB, if defined:
    if config["general"].getboolean("db_rw"):
        write_spectrogram(
            label, bboxes_df, spectrogram, spectrogram_mean, spectrogram_l2_norm, config
        )
    else:
        return bboxes_df, spectrogram, spectrogram_mean, spectrogram_l2_norm


def spect_gen_algo(config):
    """Preprocess all images

    This is a super function which provides all of the functionality to
    preprocess many wav file for model fitting. Later, this function will call
    an external algorithm to do image processing.

    Args:
        config: The parsed ini file for this particular run

    Returns:
        Nothing, write data to MongoDB w/ `db_name`

    Raises:
        Nothing
    """

    # If not writing to DB, no reason to do any preprocessing now
    if not config["general"].getboolean("db_rw"):
        return

    # Generate a Series of file names
    preprocess_files = pd.Series()
    if config["general"]["train_file"]:
        filename = f"{config['general']['data_dir']}/{config['general']['train_file']}"
        preprocess_files = preprocess_files.append(
            pd.read_csv(filename)["Filename"], ignore_index=True
        )
    if config["general"]["validate_file"]:
        filename = (
            f"{config['general']['data_dir']}/{config['general']['validate_file']}"
        )
        preprocess_files = preprocess_files.append(
            pd.read_csv(filename)["Filename"], ignore_index=True
        )
    if config["general"]["predict_file"]:
        filename = (
            f"{config['general']['data_dir']}/{config['general']['predict_file']}"
        )
        preprocess_files = preprocess_files.append(
            pd.read_csv(filename)["Filename"], ignore_index=True
        )

    # Get the number of processors
    nprocs = return_cpu_count(config)

    # Split into chunks
    chunks = np.array_split(preprocess_files, nprocs)

    # Create a ProcessPoolExecutor,
    # -> Async process everything
    executor = ProcessPoolExecutor(nprocs)
    fs = [executor.submit(chunk_preprocess, chunk, config) for chunk in chunks]
    for future in as_completed(fs):
        _ = future.result()
