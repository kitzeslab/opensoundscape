#!/usr/bin/env python3
from opensoundscape.audio import Audio, OpsoLoadAudioInputTooLong, split_and_save
import pytest
import numpy as np
from opensoundscape import signal as sig


@pytest.fixture()
def gpt_wav_str():
    return "tests/audio/great_plains_toad.wav"


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def rugr_wav_str():
    return "tests/audio/rugr_drum.wav"


def test_frequency2scale():
    """test conversion from scale -> frequency -> scale"""
    import pywt

    scale = 5
    sr = 44100
    wavelet = "morl"
    f_hz = pywt.scale2frequency(wavelet, scale) * sr
    assert sig.frequency2scale(f_hz, wavelet, sr) == scale


def test_cwt_peaks(gpt_wav_str):
    a = Audio.from_file(gpt_wav_str, sample_rate=44100).trim(5, 10)
    t, _ = sig.cwt_peaks(a, center_frequency=2500, peak_separation=0.05)
    assert len(t) == 42


def test_find_accel_sequences():
    """search for an accelerating Ruffed Grouse drumming pattern"""
    t = np.array(
        [
            1.02754281,
            1.83007625,
            1.96758198,
            2.17759073,
            3.19263303,
            3.93266386,
            4.52518855,
            5.00270845,
            5.43522647,
            5.80024168,
            5.84524355,
            6.15275636,
            6.44776866,
            6.72278012,
            6.98279095,
            7.20280012,
            7.41030876,
            7.61781741,
            7.79782491,
            7.95533147,
            8.11033793,
            8.26784449,
            8.3928497,
            8.51785491,
            8.64786033,
            8.74786449,
            8.86036918,
            8.95287304,
            9.02787616,
            9.1128797,
            9.25538564,
            9.31788825,
            9.37789075,
            9.66540273,
            9.82290929,
            9.91041293,
            10.0104171,
            10.11792158,
            10.37293221,
            13.94058086,
        ]
    )
    seq_y, seq_t = sig.find_accel_sequences(
        t,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[5, 100],
    )
    assert np.shape(seq_t) == (1, 19)


def test_detect_peak_sequence_cwt(rugr_wav_str):
    """test detection of ruffed grouse drumming

    the default parameters might change, but this should always return
    the same detection.
    """
    rugr_audio = Audio.from_file(rugr_wav_str)
    detections = sig.detect_peak_sequence_cwt(
        rugr_audio,
        sr=400,
        window_len=10,
        center_frequency=50,
        wavelet="morl",
        peak_threshold=0.2,
        peak_separation=15 / 400,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[9, 100],
        plot=False,
    )
    assert len(detections) == 1
    assert detections.iloc[0].seq_len == 24


def test_detect_peak_sequence_cwt_no_results(rugr_wav_str):
    """tests that empty dataframe is returned (instead of errror) if input audio
    is shorter than window_length 
    """
    rugr_audio = Audio.from_file(rugr_wav_str).trim(0, 1)
    detections = sig.detect_peak_sequence_cwt(
        rugr_audio,
        sr=400,
        window_len=10,
        center_frequency=50,
        wavelet="morl",
        peak_threshold=0.2,
        peak_separation=15 / 400,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[9, 100],
        plot=False,
    )
    assert len(detections) == 0


def test_detect_peak_sequence_cwt_uneven_length_results(rugr_wav_str):
    """

    this test is for the (resolved) issue #410 in which uneven lengths of
    detected sequences caused a TypeError
    """
    rugr_audio = Audio.from_file(rugr_wav_str).trim(1, 8).loop(length=20)
    detections = sig.detect_peak_sequence_cwt(
        rugr_audio,
        sr=400,
        window_len=3,
        center_frequency=50,
        wavelet="morl",
        peak_threshold=0.2,
        peak_separation=15 / 400,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[9, 100],
        plot=False,
    )
    assert len(detections) == 2
