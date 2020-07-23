#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape import ribbit
import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def gpt_path():
    return "tests/great_plains_toad.wav"


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
    t = np.linspace(0, 1, sr)
    amplitude = []
    with pytest.raises(ValueError):
        score = ribbit.calculate_pulse_score(
            amplitude,
            amplitude_sample_rate=sr,
            pulse_rate_range=[-10, 30],
            plot=False,
            nfft=1024,
        )


def test_ribbit():
    path = "./tests/silence_10s.mp3"
    audio = Audio.from_file(path, sample_rate=22050)
    spec = Spectrogram.from_audio(audio)

    scores, times = ribbit.ribbit(
        spec,
        pulse_rate_range=[5, 10],
        signal_band=[1000, 2000],
        window_len=5.0,
        noise_bands=[[0, 200]],
        plot=True,
    )
    assert len(scores) > 0


def test_pulsefinder_species_set(gpt_path):
    df = pd.DataFrame(
        columns=[
            "species",
            "pulse_rate_low",
            "pulse_rate_high",
            "low_f",
            "high_f",
            "reject_low",
            "reject_high",
            "window_length",
        ]
    )
    df.at[0, :] = ["sp1", 5, 10, 1000, 2000, 0, 500, 1.0]
    df.at[1, :] = ["sp2", 10, 15, 1000, 2000, 0, 500, 1.0]

    audio = Audio.from_file(gpt_path, sample_rate=32000)
    spec = Spectrogram.from_audio(audio, overlap_samples=256)

    df = ribbit.pulse_finder_species_set(spec, df)

    assert type(df) == pd.DataFrame


def test_summarize_top_scores(gpt_path):
    df = pd.DataFrame(
        columns=[
            "species",
            "pulse_rate_low",
            "pulse_rate_high",
            "low_f",
            "high_f",
            "reject_low",
            "reject_high",
            "window_length",
        ]
    )
    df.at[0, :] = ["sp1", 5, 10, 1000, 2000, 0, 500, 1.0]
    df.at[1, :] = ["sp2", 10, 15, 1000, 2000, 0, 500, 1.0]
    audio = Audio.from_file(gpt_path, sample_rate=32000)
    spec = Spectrogram.from_audio(audio, overlap_samples=256)
    df = ribbit.pulse_finder_species_set(spec, df)

    ribbit.summarize_top_scores(["1", "2"], [df, df], scale_factor=10.0)
