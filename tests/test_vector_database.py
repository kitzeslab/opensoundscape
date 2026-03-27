import pandas as pd
import pytest

from opensoundscape import vector_database


class _Window:
    def __init__(self, window_id, filename, start, end):
        self.id = window_id
        self.filename = filename
        self.offsets = [start, end]


def _clips_df(index_tuples):
    idx = pd.MultiIndex.from_tuples(
        index_tuples,
        names=["file", "start_time", "end_time"],
    )
    return pd.DataFrame(index=idx)


def test_normalize_index_to_tuples_rounding_and_paths():
    """Verify normalize_index_to_tuples() handles float precision and path normalization.

    This utility function converts clip indices (file, start_time, end_time) to normalized
    tuples by: (1) rounding times to avoid float precision issues, (2) normalizing file paths
    (e.g., removing double slashes). This is critical for matching clips across different
    data structures that may have slight floating-point differences.
    """
    idx = pd.MultiIndex.from_tuples(
        [
            ("a//b.wav", 0.123456789, 1.99999999),
            ("c.wav", 2.0000004, 3.0000004),
        ]
    )
    out = vector_database.normalize_index_to_tuples(idx, rounding_precision=6)
    assert out == [
        ("a/b.wav", 0.123457, 2.0),
        ("c.wav", 2.0, 3.0),
    ]


def test_find_matches_returns_positions_with_duplicates():
    A = [("a", 0.0, 1.0), ("b", 1.0, 2.0)]
    B = [("a", 0.0, 1.0), ("a", 0.0, 1.0), ("x", 2.0, 3.0)]

    matches = vector_database.find_matches(A, B)
    assert matches == [(0, 1), ()]


def test_find_matching_window_ids_preserves_clip_order(monkeypatch):
    """Verify _find_matching_window_ids() returns window IDs in the correct order.

    This function matches each clip in the input dataframe to window IDs in the database.
    The output order must correspond to the input clip order, not the database order.
    Uses monkeypatch to mock get_existing_windows() so the test doesn't require a
    real Hoplite database. Tests that windows are correctly matched even when the
    database returns them in a different order.
    """
    clips = _clips_df(
        [
            ("rec.wav", 0.0, 1.0),
            ("rec.wav", 1.0, 2.0),
            ("rec.wav", 9.0, 10.0),
        ]
    )

    windows = [
        _Window(20, "rec.wav", 1.0, 2.0),
        _Window(10, "rec.wav", 0.0, 1.0),
        _Window(11, "rec.wav", 0.0, 1.0),
    ]

    monkeypatch.setattr(
        vector_database, "get_existing_windows", lambda *args, **kwargs: windows
    )

    matched = vector_database._find_matching_window_ids(db=object(), clips=clips)
    assert matched == [(10, 11), (20,), ()]


def test_handle_existing_windows_skip_drops_existing(monkeypatch):
    """Verify _handle_existing_windows() with 'skip' mode removes clips with existing embeddings.

    When embedding_exists_mode='skip', clips that already have embeddings in the database
    should be removed from the dataframe (in-place modification). This prevents duplicate
    embeddings and allows efficient incremental updates to the database. Uses monkeypatch
    to mock the database without requiring Hoplite.
    """
    clips = _clips_df(
        [
            ("rec.wav", 0.0, 1.0),
            ("rec.wav", 1.0, 2.0),
            ("rec.wav", 2.0, 3.0),
        ]
    )

    windows = [
        _Window(1, "rec.wav", 1.0, 2.0),
        _Window(2, "rec.wav", 5.0, 6.0),
    ]

    monkeypatch.setattr(
        vector_database, "get_existing_windows", lambda *args, **kwargs: windows
    )

    vector_database._handle_existing_windows(
        db=object(),
        clips=clips,
        embedding_exists_mode="skip",
    )

    assert list(clips.index) == [
        ("rec.wav", 0.0, 1.0),
        ("rec.wav", 2.0, 3.0),
    ]


def test_handle_existing_windows_error_mode(monkeypatch):
    """Test (marked xfail) that error mode should only raise when embeddings overlap.

    When embedding_exists_mode='error', the function should raise an error only if
    *existing* embeddings conflict with the new clips being inserted (i.e., same file,
    start_time, end_time). Non-overlapping clips should not trigger an error.

    Uses monkeypatch to avoid Hoplite dependency.
    """
    clips = _clips_df(
        [
            ("rec.wav", 0.0, 1.0),
            ("rec.wav", 1.0, 2.0),
        ]
    )

    windows = [_Window(1, "other.wav", 3.0, 4.0)]
    monkeypatch.setattr(
        vector_database, "get_existing_windows", lambda *args, **kwargs: windows
    )

    vector_database._handle_existing_windows(
        db=object(),
        clips=clips,
        embedding_exists_mode="error",
    )

    # test that it raises when there is an overlap, even if floating point difference
    windows.append(_Window(2, "rec.wav", 1.0 + 1e-10, 2.0))
    monkeypatch.setattr(
        vector_database, "get_existing_windows", lambda *args, **kwargs: windows
    )
    with pytest.raises(ValueError, match="Some embeddings exist in db"):
        vector_database._handle_existing_windows(
            db=object(),
            clips=clips,
            embedding_exists_mode="error",
        )
