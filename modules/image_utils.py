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


def generate_raw_spectrogram(spectrogram, spectrogram_mean, spectrogram_l2_norm):
    """Given a normalized spectrogram

    Recreate the raw spectrogram

    Args:
        spectrogram: The normalized spectrogram
        spectrogram_mean: The np.mean(_) of the raw spectrogram
        spectrogram_l2_norm: The np.linalg.norm(_, ord=2) of the original spectrogram
    """

    # Reverse the z-score normalization
    raw_spectrogram = (spectrogram * spectrogram_l2_norm) + spectrogram_mean
    return raw_spectrogram.astype("float32", casting="same_kind")


def generate_raw_blurred_spectrogram(
    spectrogram, spectrogram_mean, spectrogram_l2_norm, gaussian_blur_sigma
):
    """Given a normalized spectrogram

    Recreate the raw spectrogram and apply a gaussian filter

    Args:
        spectrogram: The normalized spectrogram
        spectrogram_mean: The np.mean(_) of the raw spectrogram
        spectrogram_l2_norm: The np.linalg.norm(_, ord=2) of the original spectrogram
        gaussian_blur_sigma: The gaussian blur amount
    """

    # Reverse the z-score normalization
    raw_spectrogram = (spectrogram * spectrogram_l2_norm) + spectrogram_mean
    return apply_gaussian_filter(raw_spectrogram, gaussian_blur_sigma).astype(
        "float32", casting="same_kind"
    )
