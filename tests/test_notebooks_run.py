#!/usr/bin/env python3
import pytest
from opensoundscape.commands import run_command_return_code
from pathlib import Path

# This passes on OS but doesn't pass on CI
# @pytest.fixture
# def train_model_example_notebook(request):
#     base = Path("tests/notebooks/train_model_example.ipynb")
#     nbconv = Path("tests/notebooks/train_model_example.nbconvert.ipynb")
#
#     def fin():
#         # Delete downloaded wavs and directory
#         wav_path = Path("tests/notebooks/woodcock_labeled_data/")
#         labels = wav_path.joinpath("woodcock_labels.csv")
#         labels.unlink()
#         wavs = wav_path.rglob("*.wav")
#         for wav in wavs:
#             wav.unlink()
#         wav_path.rmdir()
#
#         # Delete saved results and directory
#         result_path = Path("tests/notebooks/model_train_results/")
#         results = result_path.rglob("*")
#         for result in results:
#             result.unlink()
#         result_path.rmdir()
#
#         # Delete nbconvert file
#         nbconv.unlink()
#
#     request.addfinalizer(fin)
#     return base


@pytest.fixture
def ribbit_demo_notebook(request):
    base = Path("tests/notebooks/RIBBIT_pulse_rate_demo.ipynb")
    nbconv = Path("tests/notebooks/RIBBIT_pulse_rate_demo.nbconvert.ipynb")
    yield base
    nbconv.unlink()


@pytest.fixture
def spectrogram_example_notebook(request):
    base = Path("tests/notebooks/spectrogram_example.ipynb")
    nbconv = Path("tests/notebooks/spectrogram_example.nbconvert.ipynb")
    picture_one = Path("tests/notebooks/saved_spectrogram.png")
    picture_two = Path("tests/notebooks/saved_spectrogram_2.png")
    yield base
    nbconv.unlink()
    picture_one.unlink()
    picture_two.unlink()


def check_return_code_from_notebook(notebook):
    return run_command_return_code(
        f"jupyter nbconvert --to notebook --execute {notebook}"
    )


# # This passes on OS but doesn't pass on CI
# def test_run_train_model_demo(train_model_example_notebook):
#     assert check_return_code_from_notebook(train_model_example_notebook) == 0


# def test_run_ribbit_demo(ribbit_demo_notebook):
#     assert check_return_code_from_notebook(ribbit_demo_notebook) == 0


def test_run_spectrogram_example(spectrogram_example_notebook):
    assert check_return_code_from_notebook(spectrogram_example_notebook) == 0
