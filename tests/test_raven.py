#!/usr/bin/env python3
import pytest
import opensoundscape.raven as raven
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import numpy.testing as npt
import pandas.testing as pdt

tmp_path = "tests/_tmp_raven"


@pytest.fixture()
def temporary_split_storage(request):
    path = Path(tmp_path)
    path.mkdir()
    yield path
    shutil.rmtree(path)


@pytest.fixture()
def raven_warn_dir():
    return "./tests/raven/raven_warn"


@pytest.fixture()
def raven_short_okay_dir():
    return "./tests/raven/raven_okay_short"


@pytest.fixture()
def raven_long_okay_dir():
    return "./tests/raven/raven_okay_long"


@pytest.fixture()
def raven_empty_okay_dir():
    return "./tests/raven/raven_okay_empty"


@pytest.fixture()
def raven_annots_dir():
    return "./tests/raven/raven_annots"


@pytest.fixture()
def audio_dir():
    return "./tests/audio"


@pytest.fixture()
def raven_annotations_empty(request, raven_empty_okay_dir):
    raven.lowercase_annotations(raven_empty_okay_dir)
    path = Path(f"{raven_empty_okay_dir}/EmptyExample.Table.1.selections.txt.lower")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def raven_annotations_lower_okay_short(request, raven_short_okay_dir):
    raven.lowercase_annotations(raven_short_okay_dir)
    path = Path(f"{raven_short_okay_dir}/ShortExample.Table.1.selections.txt.lower")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def raven_annotations_lower_okay_long(request, raven_long_okay_dir):
    raven.lowercase_annotations(raven_long_okay_dir)
    path = Path(f"{raven_long_okay_dir}/LongExample.Table.1.selections.txt.lower")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def raven_annotations_lower_warn(request, raven_warn_dir):
    raven.lowercase_annotations(raven_warn_dir)
    path = Path(f"{raven_warn_dir}/Example.Table.1.selections.txt.lower")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def raven_annotations_true_annots(request, raven_annots_dir):
    raven.lowercase_annotations(raven_annots_dir)
    path = Path(
        f"{raven_annots_dir}/MSD-0003_20180427_5minstart00-15.Table.1.selections.txt.lower"
    )

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


def test_raven_annotation_check_on_okay(raven_short_okay_dir):
    raven.annotation_check(raven_short_okay_dir, col="class")


def test_raven_annotation_check_on_missing_col_warns(raven_short_okay_dir):
    with pytest.warns(UserWarning):
        raven.annotation_check(raven_short_okay_dir, col="col_that_doesnt_exist")


def test_raven_annotation_check_on_missing_label_warns(raven_warn_dir):
    with pytest.warns(UserWarning):
        raven.annotation_check(raven_warn_dir, col="class")


def test_raven_lowercase_annotations_on_okay(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    assert raven_annotations_lower_okay_short.exists()


def test_raven_generate_class_corrections_with_okay(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    csv = raven.generate_class_corrections(raven_short_okay_dir, col="class")
    assert csv == "raw,corrected\nhello,hello\n"


def test_raven_generate_class_corrections_with_empty_labels(
    raven_warn_dir, raven_annotations_lower_warn
):
    csv = raven.generate_class_corrections(raven_warn_dir, col="class")
    assert csv == "raw,corrected\nunknown,unknown\n"


def test_raven_generate_class_corrections_check_on_missing_col_warns(
    raven_warn_dir, raven_annotations_lower_warn, col="class"
):
    with pytest.warns(UserWarning):
        raven.generate_class_corrections(raven_warn_dir, col="col_that_doesnt_exist")


def test_raven_query_annotations_with_okay(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    output = raven.query_annotations(raven_short_okay_dir, col="class", cls="hello")
    file_path = Path(raven_annotations_lower_okay_short)
    true_keys = [file_path]
    true_vals = pd.read_csv(file_path, sep="\t")
    assert list(output.keys()) == true_keys
    assert len(list(output.values())) == 1
    pd.testing.assert_frame_equal(list(output.values())[0], true_vals)


def test_raven_query_annotations_check_on_missing_col_warns(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    with pytest.warns(UserWarning):
        raven.query_annotations(
            raven_short_okay_dir, cls="hello", col="col_that_doesnt_exist"
        )


def test_raven_split_single_annotation_short(raven_annotations_lower_okay_short):
    result_df = raven.split_single_annotation(
        raven_annotations_lower_okay_short, col="class", split_len_s=5
    )
    pdt.assert_frame_equal(
        result_df,
        pd.DataFrame(
            {
                "seg_start": list(range(0, 381, 5)),
                "seg_end": list(range(5, 386, 5)),
                "hello": [*[0] * 71, *[1] * 6],
            }
        ),
        check_dtype=False,
    )


def test_raven_split_single_annotation_long_skiplast(raven_annotations_lower_okay_long):
    result_df = raven.split_single_annotation(
        raven_annotations_lower_okay_long, col="class", split_len_s=5
    )
    pdt.assert_frame_equal(
        result_df,
        pd.DataFrame(
            {
                "seg_start": list(range(0, 26, 5)),
                "seg_end": list(range(5, 31, 5)),
                "eato": [0, 1, 1, 1, 1, 1],
                "woth": [1, 1, 1, 1, 1, 1],
            }
        ),
        check_dtype=False,
    )


def test_raven_split_single_annotation_min_overlap(raven_annotations_lower_okay_long):
    result_df = raven.split_single_annotation(
        raven_annotations_lower_okay_long, col="class", split_len_s=5, min_label_len=1
    )
    pdt.assert_frame_equal(
        result_df,
        pd.DataFrame(
            {
                "seg_start": list(range(0, 26, 5)),
                "seg_end": list(range(5, 31, 5)),
                "eato": [0, 1, 0, 1, 0, 0],
                "woth": [1, 1, 1, 1, 1, 1],
            }
        ),
        check_dtype=False,
    )


def test_raven_split_single_annotation_long_includelast(
    raven_annotations_lower_okay_long,
):
    result_df = raven.split_single_annotation(
        raven_annotations_lower_okay_long, col="class", split_len_s=5, keep_final=True
    )
    pdt.assert_frame_equal(
        result_df,
        pd.DataFrame(
            {
                "seg_start": list(range(0, 31, 5)),
                "seg_end": list(range(5, 36, 5)),
                "eato": [0, 1, 1, 1, 1, 1, 1],
                "woth": [1, 1, 1, 1, 1, 1, 0],
            }
        ),
        check_dtype=False,
    )


def test_raven_split_single_annotation_empty(raven_annotations_empty,):
    result_df = raven.split_single_annotation(
        raven_annotations_empty, col="class", split_len_s=5
    )
    pdt.assert_frame_equal(result_df, pd.DataFrame({"seg_start": [], "seg_end": []}))


def test_raven_split_starts_ends_empty(raven_annotations_empty,):
    result_df = raven.split_starts_ends(
        raven_annotations_empty, col="class", starts=[0, 5], ends=[5, 10]
    )
    pdt.assert_frame_equal(
        result_df,
        pd.DataFrame({"seg_start": [0, 5], "seg_end": [5, 10]}),
        check_dtype=False,
    )


def test_raven_audio_split_and_save(
    temporary_split_storage, raven_annotations_true_annots, raven_annots_dir, audio_dir
):
    result_df = raven.raven_audio_split_and_save(
        raven_directory=raven_annots_dir,
        audio_directory=audio_dir,
        destination=temporary_split_storage,
        col="species",
        sample_rate=22050,
        clip_duration=5,
    )
    print(result_df.head())

    # Correct number of files created
    assert len(list(temporary_split_storage.glob("*.wav"))) == 60

    # All species found and labeled correctly
    npt.assert_array_equal(result_df.columns, ["eato", "lowa", "woth"])
    pdt.assert_frame_equal(
        pd.DataFrame([56.0, 2.0, 44.0], index=["eato", "lowa", "woth"]),
        pd.DataFrame(result_df.sum()),
    )

    # Save is same as return
    pdt.assert_frame_equal(
        result_df,
        pd.read_csv(
            temporary_split_storage.joinpath("labels.csv"), index_col="filename"
        ),
    )

    # Dataframe contains all clips
    clips_index = list(result_df.index)
    clips_index.sort()
    clips_created = [str(p) for p in temporary_split_storage.glob("*.wav")]
    clips_created.sort()
    npt.assert_array_equal(clips_index, clips_created)
