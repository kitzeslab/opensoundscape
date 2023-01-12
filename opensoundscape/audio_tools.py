"""set of tools that filter or modify audio files or sample arrays (not Audio objects)
"""
from scipy.signal import butter, sosfiltfilt


def butter_bandpass(low_f, high_f, sample_rate, order=9):
    """generate coefficients for bandpass_filter()

    Args:
        low_f: low frequency of butterworth bandpass filter
        high_f: high frequency of butterworth bandpass filter
        sample_rate: audio sample rate
        order=9: order of butterworth filter

    Returns:
        set of coefficients used in sosfiltfilt()
    """
    nyq = 0.5 * sample_rate
    low = low_f / nyq
    high = high_f / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def bandpass_filter(signal, low_f, high_f, sample_rate, order=9):
    """perform a butterworth bandpass filter on a discrete time signal
    using scipy.signal's butter and sosfiltfilt (phase-preserving filtering)

    Args:
        signal: discrete time signal (audio samples, list of float)
        low_f: -3db point (?) for highpass filter (Hz)
        high_f: -3db point (?) for highpass filter (Hz)
        sample_rate: samples per second (Hz)
        order: higher values -> steeper dropoff [default: 9]

    Returns:
        filtered time signal
    """
    sos = butter_bandpass(low_f, high_f, sample_rate, order=order)
    return sosfiltfilt(sos, signal)


def clipping_detector(samples, threshold=0.6):
    """count the number of samples above a threshold value

    Args:
        samples: a time series of float values
        threshold=0.6: minimum value of sample to count as clipping

    Returns:
        number of samples exceeding threshold
    """
    return len(list(filter(lambda x: x > threshold, samples)))
