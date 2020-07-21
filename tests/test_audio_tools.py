import pytest
from opensoundscape.audio import Audio
from opensoundscape import audio_tools
from pathlib import Path


@pytest.fixture()
def veryshort_wav_str():
    return f"tests/veryshort.wav"


@pytest.fixture()
def silent_wav_str():
    return f"tests/silence_10s.mp3"


@pytest.fixture()
def convolved_wav_str(out_path, request):
    path = Path(f"{out_path}/convolved.wav")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def out_path(request):
    path = Path("tests/audio_tools_out")
    if not path.exists(): path.mkdir()

    def fin():
        path.rmdir()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def veryshort_audio(veryshort_wav_str):
    return Audio.from_file(veryshort_wav_str)


def test_bandpass_filter(veryshort_audio):
    bandpassed = audio_tools.bandpass_filter(
        veryshort_audio.samples, 1000, 2000, veryshort_audio.sample_rate
    )
    assert len(bandpassed) == len(veryshort_audio.samples)


def test_clipping_detector(veryshort_audio):
    assert audio_tools.clipping_detector(veryshort_audio.samples) > -1


def test_silence_filter(veryshort_wav_str):
    assert audio_tools.silence_filter(veryshort_wav_str) > -1


def test_convolve_file(veryshort_wav_str, silent_wav_str, convolved_wav_str, out_path):
    audio_tools.convolve_file(silent_wav_str, convolved_wav_str, veryshort_wav_str)
    assert convolved_wav_str.exists()
