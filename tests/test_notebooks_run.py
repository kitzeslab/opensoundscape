#!/usr/bin/env python3
import pytest
from opensoundscape.commands import run_command_return_code
from pathlib import Path


@pytest.fixture
def augmentations_notebook(request):
    base = Path("notebooks/augmentations.ipynb")
    nbconv = Path("notebooks/augmentations.nbconvert.ipynb")

    def fin():
        nbconv.unlink()

    request.addfinalizer(fin)
    return base


@pytest.fixture
def pulse_finder_demo_notebook(request):
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


def test_run_augmentations(augmentations_notebook):
    assert check_return_code_from_notebook(augmentations_notebook) == 0


def test_run_pulse_finder_demo(pulse_finder_demo_notebook):
    assert check_return_code_from_notebook(pulse_finder_demo_notebook) == 0


def test_run_spectrogram_example(spectrogram_example_notebook):
    assert check_return_code_from_notebook(spectrogram_example_notebook) == 0
