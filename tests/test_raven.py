#!/usr/bin/env python3
import pytest
import opensoundscape.raven as raven
from pathlib import Path


def test_raven_annotation_check_on_okay():
    raven.annotation_check("./tests/raven_okay")


def test_raven_annotation_check_on_bad_warns():
    with pytest.warns(UserWarning):
        raven.annotation_check("./tests/raven_warn")


def test_raven_lowercase_annotations_on_okay():
    result_path = Path("./tests/raven_okay/Example.Table.1.selections.txt.lower")
    raven.lowercase_annotations("./tests/raven_okay")
    assert result_path.exists()
    result_path.unlink()


def test_raven_generate_class_corrections_with_okay():
    csv = raven.generate_class_corrections("./tests/raven_okay")
    assert csv == "raw,corrected\nhello,hello\n"


def test_raven_generate_class_corrections_with_bad():
    csv = raven.generate_class_corrections("./tests/raven_warn")
    assert csv == "raw,corrected\nunknown,unknown\n"
