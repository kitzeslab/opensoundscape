import pandas as pd
import numpy as np
from scipy import signal, ndimage
from scipy.io import wavfile
from scikits.samplerate import resample
from skimage.morphology import remove_small_objects
from modules.db_utils import write_spectrogram

def generate_segments_from_binary_spectrogram(binary_spec, buffer):
    '''Identify feature bounding boxes

    Given a binary spectrogram, label the segments and find the bounding
    boxes (plus a buffer) around each feature segment.
    
    Args:
        binary_spec: A binary spectrogram
        buffer: A number of pixels to add to the bounding boxes

    Returns:
        A dataframe containing the bounding boxes
    '''
    # Label the segments and get raw bounding boxes
    labeled_segments, num_of_segments = ndimage.label(binary_spec)
    raw_bboxes = [ndimage.find_objects(labeled_segments == x)
            for x in range(1, num_of_segments + 1)]

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
    '''Filter via scaled median threshold

    Given a spectrogram, filter rows and columns by the median with a scaling
    factor anything below the threshold is set to 0.0. _filter_by_scaled_median
    is the function which allows us to do this.  By returning the logical_and
    we are converting the spectrogram to binary for additional processing.

    Args:
        spec: The spectrogram generated from signal.spectrogram
        factor: The factor to pass to the median filter

    Returns:
        A binary spectrogram
    '''

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


def resample_audio(samples, current_sample_rate, new_sample_rate):
    '''Resamples audio samples

    Given an audio sample and the current sample rate, resample to a new
    sample rate.

    Args:
        samples: The audio samples from wavfile.read(<file>)[0]
        current_sample_rate: The current sample rate from wavfile.read(<file>)[1]
        new_sample_rate: The new sample rate

    Returns:
        A tuple with the new sample rate and the new audio samples. This
        is exactly similar to wavfile.read(<filename>)
    '''
    return new_sample_rate, resample(samples,
            new_sample_rate/current_sample_rate, 'sinc_best')


def frequency_based_spectrogram_filter(spec, freq, low_freq_thr, high_freq_thr):
    '''Filter any useless low and high frequency ranges

    Given a spectrogram and frequencies generated from signal.spectrogram
    filter any low and high frequencies we aren't interested in analyzing.

    Args:
        spec: The input spectrogram from signal.spectrogram
        freq: The input frequencies from signal.spectrogram
        low_freq_thr: The low frequency threshold
        high_freq_thr: The high frequency threshold

    Returns:
        A tuple containing the new spectrogram and frequencies
    '''
    indices = np.argwhere((freq > low_freq_thr) & (freq < high_freq_thr)).flatten()
    return spec[indices, :], freq[indices]


def spect_gen(file, config):
    '''Preprocess a wav file
    
    This is a super function which provides all of the functionality to
    preprocess a wav file for our template matching procedure. Later,
    this function will call an external algorithm to do image processing.

    Args:
        file: A wav file for template matching
        config: The parsed ini file for this particular run

    Returns:
        - If `db_readwrite = False`, write data to MongoDB w/ `db_name` &
          `db_collection_name`
        - Else returns a tuple containing: bounding boxes as a DataFrame,
          spectrogram (numpy 2D matrix), and the normalization factor

    Raises:
        Nothing
    '''

    # Resample
    sample_rate, samples = wavfile.read("{}/{}".format(config['data_dir'], file))
    sample_rate, samples = resample_audio(samples, sample_rate,
            config.getfloat('resample_rate'))

    # Generate Spectrogram
    nperseg = config.getint('spectrogram_segment_length')
    noverlap = int(nperseg * config.getfloat('spectrogram_overlap') / 100.)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate,
            window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nperseg)

    # Frequency Selection
    spectrogram, frequencies = frequency_based_spectrogram_filter(spectrogram,
            frequencies, config.getint('low_freq_thresh'),
            config.getint('high_freq_thresh'))

    # Normalization, need spectrogram_max later
    spectrogram_max = spectrogram.max()
    spectrogram /= spectrogram_max

    # Scaled Median Column/Row Filters
    binary_spectrogram = scaled_median_filter(spectrogram,
            config.getfloat('median_filter_factor'))

    # Binary Closing
    binary_spectrogram = ndimage.morphology.binary_closing(binary_spectrogram,
            structure=np.ones((
                config.getint('binary_closing_kernel_height'),
                config.getint('binary_closing_kernel_width'))))

    # Binary Dilation
    binary_spectrogram = ndimage.morphology.binary_dilation(binary_spectrogram,
            structure=np.ones((
                config.getint('binary_dilation_kernel_height'),
                config.getint('binary_dilation_kernel_width'))))

    # Binary Median Filter
    binary_spectrogram = ndimage.median_filter(binary_spectrogram,
            size=(
                config.getint('median_filter_kernel_height'),
                config.getint('median_filter_kernel_width')))

    # Remove Small Objects
    binary_spectrogram = remove_small_objects(binary_spectrogram,
            config.getint('small_objects_kernel_size'))

    # Get the bounding box dataframe from the labeled segments
    bboxes_df = generate_segments_from_binary_spectrogram(binary_spectrogram,
            config.getint('segment_pixel_buffer'))

    # Finally store the data
    if config.getboolean('db_readwrite'):
        write_spectrogram(file, bboxes_df, spectrogram, spectrogram_max,
                config)
    else:
        return bboxes_df, spectrogram, float(spectrogram_max)
