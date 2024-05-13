#!/usr/bin/env python3
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape import ribbit
import pytest
import numpy as np
import pandas as pd
import math


@pytest.fixture()
def gpt_path():
    return "tests/audio/great_plains_toad.wav"


@pytest.fixture()
def veryshort_wav_str():
    return "tests/audio/veryshort.wav"


def test_calculate_pulse_score():
    sr = 100
    t = np.linspace(0, 1, sr)
    amplitude = np.sin(t * 2 * np.pi * 20)
    score = ribbit.calculate_pulse_score(
        amplitude,
        amplitude_sample_rate=sr,
        pulse_rate_range=[0, 30],
        plot=False,
        nfft=1024,
    )
    assert score > 0


def test_calculate_pulse_score_zero_len_input():
    sr = 100
    amplitude = []
    with pytest.raises(ValueError):
        _ = ribbit.calculate_pulse_score(
            amplitude,
            amplitude_sample_rate=sr,
            pulse_rate_range=[-10, 30],
            plot=False,
            nfft=1024,
        )


def test_ribbit(gpt_path):
    audio = Audio.from_file(gpt_path, sample_rate=22050).trim(0, 16)

    spec = Spectrogram.from_audio(
        audio,
        window_samples=512,
        overlap_samples=256,
    )

    df = ribbit.ribbit(
        spec,
        pulse_rate_range=[5, 10],
        signal_band=[1000, 2000],
        clip_duration=5.0,
        clip_overlap=0,
        final_clip=None,
        noise_bands=[[0, 200]],
        plot=False,
    )

    assert len(df) == 3
    assert math.isclose(max(df["score"]), 0.0392323, abs_tol=1e-4)


def test_ribbit_short_audio(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    spec = Spectrogram.from_audio(
        audio,
        window_samples=512,
        overlap_samples=256,
    )

    df = ribbit.ribbit(
        spec,
        pulse_rate_range=[5, 10],
        signal_band=[1000, 2000],
        clip_duration=5.0,
        clip_overlap=2.5,
        final_clip=None,
        noise_bands=[[0, 200]],
        plot=False,
    )
    assert len(df) == 0


def test_ribbit_high_spec_overlap(gpt_path):
    """spec params should not effect number of clips in results"""
    audio = Audio.from_file(gpt_path, sample_rate=22050).trim(0, 16)
    spec = Spectrogram.from_audio(audio, window_samples=512, overlap_samples=500)

    df = ribbit.ribbit(
        spec,
        pulse_rate_range=[5, 10],
        signal_band=[1000, 2000],
        clip_duration=5.0,
        clip_overlap=0,
        final_clip=None,
        noise_bands=[[0, 200]],
        plot=False,
    )
    assert len(df) == 3
    assert math.isclose(max(df["start_time"]), 10.0, abs_tol=1e-4)


def test_ribbit_with_clip_overlap(gpt_path):
    audio = Audio.from_file(gpt_path, sample_rate=22050).trim(0, 16)

    spec = Spectrogram.from_audio(audio, window_samples=512, overlap_samples=256)

    df = ribbit.ribbit(
        spec,
        pulse_rate_range=[5, 10],
        signal_band=[1000, 2000],
        clip_duration=5.0,
        clip_overlap=2.5,
        final_clip=None,
        noise_bands=[[0, 200]],
        plot=False,
    )

    assert len(df) == 5
    assert math.isclose(max(df["score"]), 0.039380, abs_tol=1e-4)
