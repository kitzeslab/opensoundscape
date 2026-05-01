import datetime
import sys
import types

import numpy as np
import pandas as pd
import pytest

from opensoundscape import vector_database


class _Window:
    def __init__(self, window_id, filename, start, end):
        self.id = window_id
        self.filename = filename
        self.offsets = [start, end]


class _Recording:
    def __init__(self, record_id, filename, deployment_id=1, dt=None):
        self.id = record_id
        self.filename = filename
        self.deployment_id = deployment_id
        self.datetime = dt


class _Deployment:
    def __init__(self, dep_id, name, project):
        self.id = dep_id
        self.name = name
        self.project = project


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
    """Verify find_matches() returns all matching positions per tuple in order."""
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


def test_normalize_windows_to_tuples_rounding_and_path_cleanup():
    """Verify normalize_windows_to_tuples() rounds offsets and normalizes filename paths."""
    windows = [
        _Window(1, "root//a.wav", 0.123456789, 1.9999999),
        _Window(2, "b.wav", 2.0, 3.0000004),
    ]
    out = vector_database.normalize_windows_to_tuples(windows)
    assert out == [("root/a.wav", 0.123457, 2.0), ("b.wav", 2.0, 3.0)]


def test_build_index_groups_positions_per_tuple():
    """Verify build_index() maps each tuple to every index where it appears."""
    items = [("a", 0, 1), ("b", 1, 2), ("a", 0, 1)]
    idx = vector_database.build_index(items)
    assert idx[("a", 0, 1)] == [0, 2]
    assert idx[("b", 1, 2)] == [1]


def test_get_existing_windows_assigns_filenames(monkeypatch):
    """Verify get_existing_windows() adds filename to each returned window from recording IDs."""

    class FakeDB:
        def __init__(self):
            self.recordings = [_Recording(7, "rec.wav")]

        def get_all_recordings(self):
            return self.recordings

        def match_window_ids(self, recordings_filter=None, deployments_filter=None):
            return [100]

        def get_all_windows(self, filter=None):
            win = types.SimpleNamespace(id=100, recording_id=7, offsets=[0.0, 1.0])
            return [win]

    monkeypatch.setattr(vector_database, "_require_hoplite", lambda: None)
    windows = vector_database.get_existing_windows(FakeDB(), ["rec.wav"])
    assert len(windows) == 1
    assert windows[0].filename == "rec.wav"


def test_insert_embeddings_inserts_and_tracks_duplicates_and_clip_overflow():
    """Verify _insert_embeddings() inserts recordings/windows and reports duplicate windows."""

    class FakeDB:
        def __init__(self):
            self._next_recording = 10
            self.inserted_recordings = []
            self.inserted_windows = []

        def get_embedding_dtype(self):
            return "float16"

        def insert_recording(self, filename, deployment_id=None, datetime=None):
            rid = self._next_recording
            self._next_recording += 1
            self.inserted_recordings.append((rid, filename, deployment_id, datetime))
            return rid

        def insert_window(self, recording_id, embedding, offsets):
            key = (recording_id, offsets[0], offsets[1])
            if key in {(10, 1.0, 2.0)}:
                raise RuntimeError("Duplicate key")
            self.inserted_windows.append((recording_id, embedding, offsets))

    sample_a = types.SimpleNamespace(source="new.wav", start_time=0.0, duration=1.0)
    sample_b = types.SimpleNamespace(source="new.wav", start_time=1.0, duration=1.0)
    db = FakeDB()
    file_to_id = {}
    embs = np.array([[1e9, -1e9], [0.1, 0.2]], dtype=np.float32)

    window_ids, failures = vector_database._insert_embeddings(
        db=db,
        batch_samples=[sample_a, sample_b],
        batch_embeddings=embs,
        overflow_mode="clip",
        file_to_id=file_to_id,
    )

    assert file_to_id["new.wav"] == 10
    assert len(db.inserted_windows) == 1
    inserted_emb = db.inserted_windows[0][1]
    assert inserted_emb.dtype == np.float16
    assert np.isfinite(inserted_emb).all()
    assert failures == [("new.wav", 1.0, 2.0, "duplicate window")]


def test_collate_search_results_returns_window_metadata():
    """Verify _collate_search_results() joins search hits with db metadata fields."""

    class FakeDB:
        def get_window(self, window_id):
            return types.SimpleNamespace(
                id=window_id, recording_id=5, offsets=[2.0, 3.5]
            )

        def get_recording(self, recording_id):
            return types.SimpleNamespace(filename="clip.wav")

    matches = [
        types.SimpleNamespace(window_id=12, sort_score=0.9),
        types.SimpleNamespace(window_id=13, sort_score=0.8),
    ]
    results = types.SimpleNamespace(search_results=matches)
    out = vector_database._collate_search_results(FakeDB(), results)
    assert out[0]["file"] == "clip.wav"
    assert out[0]["window_id"] == 12
    assert out[1]["sort_score"] == 0.8


def test_remove_duplicate_windows_removes_repeated_offsets_for_same_recording():
    """Verify remove_duplicate_windows() removes only duplicate time windows per recording."""

    class FakeDB:
        def __init__(self):
            self.removed = []
            self.windows = [
                types.SimpleNamespace(id=1, recording_id=3, offsets=[0.0, 1.0]),
                types.SimpleNamespace(id=2, recording_id=3, offsets=[0.0, 1.0]),
                types.SimpleNamespace(id=3, recording_id=3, offsets=[1.0, 2.0]),
            ]

        def get_all_windows(self, include_embedding=False):
            return self.windows

        def get_recording(self, recording_id):
            return types.SimpleNamespace(filename="rec.wav")

        def remove_window(self, window_id):
            self.removed.append(window_id)

        def count_embeddings(self):
            return 2

    db = FakeDB()
    vector_database.remove_duplicate_windows(db)
    assert db.removed == [2]


def test_check_or_set_model_id_sets_missing_and_rejects_mismatch():
    """Verify _check_or_set_model_id() writes missing model_id and rejects mismatched IDs."""

    class FakeDB:
        def __init__(self):
            self.metadata = {}

        def get_metadata(self, _):
            return self.metadata

        def insert_metadata(self, key, value):
            self.metadata[key] = dict(value)

    db = FakeDB()
    vector_database._check_or_set_model_id(db, "model-a")
    assert db.metadata["model"]["model_id"] == "model-a"

    with pytest.raises(ValueError, match="Model ID mismatch"):
        vector_database._check_or_set_model_id(db, "model-b")


def test_check_or_set_sample_duration_sets_missing_and_rejects_mismatch():
    """Verify _check_or_set_sample_duration() writes missing value and rejects mismatches."""

    class FakeDB:
        def __init__(self):
            self.metadata = {}

        def get_metadata(self, _):
            return self.metadata

        def insert_metadata(self, key, value):
            self.metadata[key] = dict(value)

    db = FakeDB()
    vector_database._check_or_set_sample_duration(db, 5.0)
    assert db.metadata["model"]["sample_duration"] == 5.0

    with pytest.raises(ValueError, match="Sample duration mismatch"):
        vector_database._check_or_set_sample_duration(db, 3.0)


def test_require_hoplite_import_succeeds_when_package_installed():
    """Verify _require_hoplite() succeeds when hoplite dependencies are available."""
    pytest.importorskip("perch_hoplite")
    vector_database._require_hoplite()


def test_find_matching_windows_builds_filters_and_applies_time_range(monkeypatch):
    """Verify find_matching_windows() returns windows annotated with recording/deployment metadata."""

    class FakeDB:
        def __init__(self):
            self.recording = _Recording(
                10,
                "rec.wav",
                deployment_id=21,
                dt=datetime.datetime(2025, 1, 1, 8, 30, 0),
            )

        def match_window_ids(
            self,
            deployments_filter=None,
            recordings_filter=None,
            windows_filter=None,
            annotations_filter=None,
        ):
            return [33]

        def get_window(self, window_id):
            return types.SimpleNamespace(id=window_id, recording_id=10, offsets=[0, 5])

        def get_recording(self, recording_id):
            return self.recording

        def get_deployment(self, deployment_id):
            return _Deployment(21, "dep-a", "proj-a")

    monkeypatch.setattr(vector_database, "_require_hoplite", lambda: None)

    wins = vector_database.find_matching_windows(
        FakeDB(),
        deployments=["dep-a"],
        projects=["proj-a"],
        recordings=["rec.wav"],
        date_range=("2025-01-01", "2025-12-31"),
        time_range=("08:00:00", "09:00:00"),
    )

    assert len(wins) == 1
    assert wins[0].filename == "rec.wav"
    assert wins[0].deployment == "dep-a"
    assert wins[0].project == "proj-a"


def test_windows_to_dataframe_exports_expected_columns():
    """Verify windows_to_dataframe() creates a dataframe with expected metadata columns."""
    window = types.SimpleNamespace(
        filename="f.wav",
        offsets=[0.0, 2.0],
        datetime=datetime.datetime(2025, 1, 1, 0, 0, 0),
        deployment="dep",
        project="proj",
        id=7,
    )
    df = vector_database.windows_to_dataframe([window])
    assert list(df.columns) == [
        "file",
        "start_time",
        "end_time",
        "datetime",
        "deployment",
        "project",
        "window_id",
    ]
    assert df.iloc[0]["window_id"] == 7


def test_similarity_search_hoplite_db_ann_mode(monkeypatch):
    """Verify ANN mode wraps db.ui.search hits into collated window metadata results."""

    class FakeTopK:
        def __init__(self, top_k):
            self.search_results = []

        def update(self, search_result):
            self.search_results.append(search_result)

    class FakeSearchResult:
        def __init__(self, window_id, sort_score):
            self.window_id = window_id
            self.sort_score = sort_score

    fake_brutalism = types.SimpleNamespace(threaded_brute_search=None)
    fake_score_functions = types.SimpleNamespace(
        get_score_fn=lambda *args, **kwargs: None
    )
    fake_search_results = types.SimpleNamespace(
        TopKSearchResults=FakeTopK,
        SearchResult=FakeSearchResult,
    )

    monkeypatch.setitem(sys.modules, "perch_hoplite", types.ModuleType("perch_hoplite"))
    monkeypatch.setitem(
        sys.modules, "perch_hoplite.db", types.ModuleType("perch_hoplite.db")
    )
    monkeypatch.setitem(sys.modules, "perch_hoplite.db.brutalism", fake_brutalism)
    monkeypatch.setitem(
        sys.modules, "perch_hoplite.db.score_functions", fake_score_functions
    )
    monkeypatch.setitem(
        sys.modules, "perch_hoplite.db.search_results", fake_search_results
    )

    class FakeANN:
        keys = [1, 2]
        distances = [0.8, 0.7]

    class FakeDB:
        def __init__(self):
            self.ui = types.SimpleNamespace(
                search=lambda emb, count, **kwargs: FakeANN()
            )

        def get_embedding_dtype(self):
            return np.float32

    monkeypatch.setattr(
        vector_database,
        "_collate_search_results",
        lambda db, results: [{"window_id": 1}, {"window_id": 2}],
    )

    out = vector_database.similarity_search_hoplite_db(
        query_embedding=np.array([0.1, 0.2], dtype=np.float64),
        db=FakeDB(),
        num_results=2,
        exact_search=False,
    )
    assert out == [{"window_id": 1}, {"window_id": 2}]


def test_similarity_search_hoplite_db_rejects_subset_for_ann_mode():
    """Verify ANN mode raises when exact-search-only arguments are supplied."""

    class DummyDB:
        ui = types.SimpleNamespace(search=lambda *args, **kwargs: None)

        def get_embedding_dtype(self):
            return np.float32

    with pytest.raises(NotImplementedError, match="search_subset_size"):
        vector_database.similarity_search_hoplite_db(
            np.array([1.0], dtype=np.float32),
            DummyDB(),
            exact_search=False,
            search_subset_size=10,
        )


def test_load_or_create_hoplite_usearch_db_accepts_hoplite_interface_object(
    monkeypatch,
):
    """Verify load_or_create_hoplite_usearch_db() returns pre-initialized HopliteDB objects."""
    fake_interface_module = types.ModuleType("perch_hoplite.db.interface")

    class FakeInterface:
        pass

    fake_interface_module.HopliteDBInterface = FakeInterface
    fake_perch = types.ModuleType("perch_hoplite")
    fake_db_mod = types.ModuleType("perch_hoplite.db")
    fake_db_mod.interface = fake_interface_module
    fake_perch.db = fake_db_mod

    monkeypatch.setitem(sys.modules, "perch_hoplite", fake_perch)
    monkeypatch.setitem(sys.modules, "perch_hoplite.db", fake_db_mod)
    monkeypatch.setitem(
        sys.modules, "perch_hoplite.db.interface", fake_interface_module
    )

    sqlite_mod = types.ModuleType("perch_hoplite.db.sqlite_usearch_impl")
    sqlite_mod.SQLiteUSearchDB = types.SimpleNamespace(
        create=lambda *args, **kwargs: None
    )
    sqlite_mod.get_default_usearch_config = lambda dim: {"embedding_dim": dim}
    monkeypatch.setitem(sys.modules, "perch_hoplite.db.sqlite_usearch_impl", sqlite_mod)
    monkeypatch.setattr(vector_database, "_require_hoplite", lambda: None)

    db = FakeInterface()
    out = vector_database.load_or_create_hoplite_usearch_db(db)
    assert out is db
