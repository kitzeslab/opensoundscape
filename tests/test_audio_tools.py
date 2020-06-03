import pytest
from os.path import exists
import os
from opensoundscape import audio_tools
from opensoundscape.audio import Audio

# audio_tools.run_command('rm ./audio_tools_out/*')

@pytest.fixture()
def veryshort_wav_str():
    return f"tests/veryshort.wav"

@pytest.fixture()
def silent_wav_str():
    return f"tests/silence_10s.mp3"

@pytest.fixture()
def convolved_wav_str():
    return f"tests/audio_tools_out/convolved.wav"

@pytest.fixture()
def out_path():
    return f"tests/audio_tools_out"

audio = Audio(f"tests/veryshort.wav")

# def test_audio_gate(veryshort_wav_str):
#     os.system('pwd')
#     print(audio_tools.audio_gate(veryshort_wav_str,'./tests/audio_tools_out/gated.wav'))
#     assert(exists('./tests/audio_tools_out/gated.wav'))
    
def test_bandpass_filter():
    bandpassed = audio_tools.bandpass_filter(audio.samples,1000,2000,audio.sample_rate)
    assert(len(bandpassed)==len(audio.samples))

def test_clipping_detector():
    assert(audio_tools.clipping_detector(audio.samples)>-1)

def test_silence_filter(veryshort_wav_str):
    assert(audio_tools.silence_filter(veryshort_wav_str)>-1)
    
def test_convolve_file(veryshort_wav_str,silent_wav_str,convolved_wav_str,out_path):
    if not exists(out_path):
        os.system(f'mkdir {outpath}')
    if exists(convolved_wav_str):
        os.system(f'rm {convolved_wav_str}')
    audio_tools.convolve_file(silent_wav_str, convolved_wav_str, veryshort_wav_str)
    assert(exists(convolved_wav_str))