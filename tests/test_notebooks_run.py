#!/usr/bin/env python3
import pytest
from opensoundscape.commands import run_command_return_code
from pathlib import Path

# This passes on OS but doesn't pass on CI
# @pytest.fixture
# def train_model_example_notebook(request):
#     base = Path("notebooks/train_model_example.ipynb")
#     nbconv = Path("notebooks/train_model_example.nbconvert.ipynb")
#
#     def fin():
#         # Delete downloaded wavs and directory
#         wav_path = Path("notebooks/woodcock_labeled_data/")
#         labels = wav_path.joinpath("woodcock_labels.csv")
#         labels.unlink()
#         wavs = wav_path.rglob("*.wav")
#         for wav in wavs:
#             wav.unlink()
#         wav_path.rmdir()
#
#         # Delete saved results and directory
#         result_path = Path("notebooks/model_train_results/")
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
    base = Path("notebooks/RIBBIT_pulse_rate_demo.ipynb")
    nbconv = Path("notebooks/RIBBIT_pulse_rate_demo.nbconvert.ipynb")

    def fin():
        nbconv.unlink()

    request.addfinalizer(fin)
    return base


@pytest.fixture
def spectrogram_example_notebook(request):
    base = Path("notebooks/spectrogram_example.ipynb")
    nbconv = Path("notebooks/spectrogram_example.nbconvert.ipynb")
    picture_one = Path("notebooks/saved_spectrogram.png")
    picture_two = Path("notebooks/saved_spectrogram_2.png")

    def fin():
        nbconv.unlink()
        picture_one.unlink()
        picture_two.unlink()

    request.addfinalizer(fin)
    return base


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
