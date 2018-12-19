from scipy.ndimage.filters import gaussian_filter


def apply_gaussian_filter(spec, sigma):
    """Apply a Gaussian filter to a spectrogram

    Apply a Gaussian filter to a spectrogram with a given sigma value.

    Args:
        spec: The input spectrogram
        sigma: The filter sigma value

    Returns:
        The Gaussian filtered spectrogram
    """
    return gaussian_filter(spec, sigma=float(sigma))


def generate_raw_blurred_spectrogram(
    spectrogram, normalization_factor, gaussian_blur_sigma
):
    """Given a normalized spectrogram

    Recreate the raw spectrogram and apply a gaussian filter

    Args:
        spectrogram: The normalized spectrogram
        normalization_factor: The factor to multiply the spectrogram
        gaussian_blur_sigma: The gaussian blur amount
    """

    raw_spectrogram = spectrogram * normalization_factor
    return apply_gaussian_filter(raw_spectrogram, gaussian_blur_sigma).astype(
        "float32", casting="same_kind"
    )
