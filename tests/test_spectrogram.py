#!/usr/bin/env python3
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram, MelSpectrogram
import pytest
import numpy as np
from math import isclose


@pytest.fixture()
def veryshort_wav_str():
    return "tests/audio/veryshort.wav"


def test_spectrogram_raises_typeerror():
    with pytest.raises(TypeError):
        Spectrogram.from_audio("not samples")


def test_spectrogram_shape_of_veryshort(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    spec = Spectrogram.from_audio(audio, overlap_samples=384)
    assert spec.spectrogram.shape == (257, 21)
    assert spec.frequencies.shape == (257,)
    assert spec.times.shape == (21,)
    assert isclose(spec.window_length(), 0.02321995465, abs_tol=1e-4)
    assert isclose(spec.window_step(), 0.005804988662, abs_tol=1e-4)
    assert isclose(spec.duration(), audio.duration(), abs_tol=1e-2)
    assert isclose(spec.window_start_times()[0], 0, abs_tol=1e-4)


def test_spectrogram_shape_of_windowlengths_overlapfraction(veryshort_wav_str):
    # test spectrogram construction using window length in s and overlap fraction
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    spec = Spectrogram.from_audio(
        audio, window_length_sec=0.02321995465, overlap_fraction=0.75
    )

    assert spec.spectrogram.shape == (257, 21)
    assert spec.frequencies.shape == (257,)
    assert spec.times.shape == (21,)


def test_construct_spectrogram_spectrogram_str_raises():
    with pytest.raises(TypeError):
        Spectrogram("raises", np.zeros((5)), np.zeros((10)), (-100, -20))


def test_construct_spectrogram_frequencies_str_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), "raises", np.zeros((10)), (-100, -20))


def test_construct_spectrogram_times_str_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), "raises", (-100, -20))


def test_construct_spectrogram_spectrogram_1d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5)), np.zeros((5)), np.zeros((5)), (-100, -20))


def test_construct_spectrogram_frequencies_2d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5)), (-100, -20))


def test_construct_spectrogram_times_2d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 5)), np.zeros((5)), np.zeros((5, 5)), (-100, -20))


def test_construct_spectrogram_dimensions_mismatch_raises_one():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((7)), (-100, -20))


def test_construct_spectrogram_dimensions_mismatch_raises_two():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((3)), np.zeros((10)), (-100, -20))


def test_construct_spectrogram_no_decibel_limits_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)))


def test_construct_spectrogram_decibel_limits_incorrect_dimensions_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100))


def test_construct_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100, -20))


def test_bandpass_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10), (-100, -20)
    ).bandpass(2, 4)


def test_bandpass_spectrogram_out_of_bounds():
    with pytest.raises(ValueError):
        Spectrogram(
            np.zeros((5, 10)),
            np.linspace(0, 10, 5),
            np.linspace(0, 10, 10),
            (-100, -20),
        ).bandpass(0, 11, out_of_bounds_ok=False)


def test_bandpass_spectrogram_not_out_of_bounds():
    Spectrogram(
        np.zeros((5, 10)), np.linspace(0, 10, 5), np.linspace(0, 10, 10), (-100, -20)
    ).bandpass(0.0, 10.0, out_of_bounds_ok=False)


def test_bandpass_spectrogram_bad_limits():
    with pytest.raises(ValueError):
        Spectrogram(
            np.zeros((5, 10)),
            np.linspace(0, 100, 5),
            np.linspace(0, 10, 10),
            (-100, -20),
        ).bandpass(4, 2)


def test_trim_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10), (-100, -20)
    ).trim(2, 4)


def test_limit_db_range():
    s = Spectrogram(
        np.random.normal(0, 200, [5, 10]), np.zeros((5)), np.zeros((10)), (-100, -20)
    ).limit_db_range(-100, -20)
    assert np.max(s.spectrogram) <= -20 and np.min(s.spectrogram) >= -100


def test_plot_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100, -20)).plot()


def test_amplitude_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100, -20)
    ).amplitude()


def test_net_amplitude_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10), (-100, -20)
    ).net_amplitude([50, 100], [[0, 10], [20, 30]])


def test_to_image():
    from PIL.Image import Image

    print(
        type(
            Spectrogram(
                np.zeros((5, 10)),
                np.linspace(0, 100, 5),
                np.linspace(0, 10, 10),
                (-100, -20),
            ).to_image()
        )
    )
    assert isinstance(
        Spectrogram(
            np.zeros((5, 10)),
            np.linspace(0, 100, 5),
            np.linspace(0, 10, 10),
            (-100, -20),
        ).to_image(),
        Image,
    )


def test_to_image_with_bandpass():
    from PIL.Image import Image

    print(
        type(
            Spectrogram(
                np.zeros((5, 10)),
                np.linspace(0, 100, 5),
                np.linspace(0, 10, 10),
                (-100, -20),
            ).to_image()
        )
    )
    assert isinstance(
        Spectrogram(
            np.zeros((5, 10)),
            np.linspace(0, 100, 5),
            np.linspace(0, 10, 10),
            (-100, -20),
        ).to_image(),
        Image,
    )


def test_melspectrogram_shape_of_S_for_veryshort(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    assert mel_spec.spectrogram.shape == (64, 11)


def test_melspectrogram_to_image_works(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    assert mel_spec.to_image()


def test_melspectrogram_to_image_numchannels(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    img = mel_spec.to_image(shape=(10, 20), channels=4)
    assert img.size == (20, 10)
    arr = np.array(img)
    assert arr.shape == (10, 20, 4)


def test_melspectrogram_to_image_alltypes(veryshort_wav_str):
    from PIL.Image import Image
    from torch import Tensor

    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    img = mel_spec.to_image(shape=(10, 20), return_type="pil")
    assert isinstance(img, Image)
    img = mel_spec.to_image(shape=(10, 20), return_type="np")
    assert isinstance(img, np.ndarray)
    img = mel_spec.to_image(shape=(10, 20), return_type="torch")
    assert isinstance(img, Tensor)


def test_melspectrogram_to_image_with_invert(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    positive = mel_spec.to_image(shape=(10, 20), invert=False, return_type="np")
    negative = mel_spec.to_image(shape=(10, 20), invert=True, return_type="np")
    assert np.array_equal(negative, 1 - positive)


def test_melspectrogram_trim_works(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio).trim(0, 1)


def test_melspectrogram_bandpass_works(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio).bandpass(2000, 3000)
    assert mel_spec.spectrogram.shape == (8, 11)
