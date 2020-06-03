#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.pulse_finder import *
import pytest
import numpy as np
import pandas as pd

print("running")

# @pytest.fixture()


def test_pulse_finder():
    path = "./tests/silence_10s.mp3"
    audio = Audio(path)
    spec = Spectrogram.from_audio(audio)

    scores, times = pulse_finder(
        spec,
        pulse_rate_range=[5, 10],
        freq_range=[1000, 2000],
        window_len=5.0,
        rejection_bands=[[0, 200]],
        plot=True,
    )
    assert len(scores) > 0


def test_pulsefinder_species_set():
    path = "./tests/great_plains_toad.wav"
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

    audio = Audio(path, sample_rate=32000)
    spec = Spectrogram.from_audio(audio, overlap_samples=256)

    df = pulse_finder_species_set(spec, df)

    assert type(df) == pd.DataFrame
