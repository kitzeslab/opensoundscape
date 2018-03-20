from scipy.ndimage.filters import gaussian_filter


def apply_gaussian_filter(spec, sigma):
    '''Apply a Gaussian filter to a spectrogram

    Apply a Gaussian filter to a spectrogram with a given sigma value.

    Args:
        spec: The input spectrogram
        sigma: The filter sigma value

    Returns:
        The Gaussian filtered spectrogram
    '''
    return gaussian_filter(spec, sigma=sigma)
