import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from opensoundscape.ml import song_space
from opensoundscape.ml.song_space import SongSpace


class _FakeDB:
    def __init__(self, dim=4):
        self._dim = dim

    def get_embedding_dim(self):
        return self._dim

    def get_embeddings_batch(self, ids):
        # return a deterministic array for testing
        return np.vstack([np.arange(self._dim) + i for i in range(len(ids))]).astype(
            np.float32
        )

    def get_all_deployments(self):
        return []

    def get_all_recordings(self):
        return []


def make_songspace(tmp_path, monkeypatch, embed_dim=4):
    # Fake feature extractor object with classifier.in_features and sample_duration
    fe = SimpleNamespace()
    fe.classifier = SimpleNamespace(in_features=embed_dim)
    fe.sample_duration = 1.0
    # provide a minimal similarity_search_hoplite_db to allow delegation tests
    fe.similarity_search_hoplite_db = lambda *args, **kwargs: "SIMILAR"

    # monkeypatch the DB factory to return a lightweight fake DB
    monkeypatch.setattr(
        song_space,
        "load_or_create_hoplite_usearch_db",
        lambda path, embedding_dim: _FakeDB(embedding_dim),
    )

    ss = SongSpace(path=str(tmp_path / "ss"), feature_extractor=fe)
    return ss


def test_add_remove_and_list_classifiers(tmp_path, monkeypatch):
    ss = make_songspace(tmp_path, monkeypatch)

    # adding a model without 'classes' should raise
    with pytest.raises(ValueError):
        ss.add_classifier("bad", object())

    # add a proper classifier
    model = SimpleNamespace(classes=["a", "b"])
    ss.add_classifier("good", model)
    assert "good" in ss.list_classifiers()

    ss.remove_classifier("good")
    assert "good" not in ss.list_classifiers()


def test_metrics_computes_and_handles_nans():
    # simple preds/labels with one class fully NaN and one valid
    preds = pd.DataFrame({"c1": [0.1, 0.9, 0.8], "c2": [0.2, 0.3, 0.4]})
    labels = pd.DataFrame({"c1": [0, 1, 1], "c2": [np.nan, np.nan, np.nan]})
    ss = SimpleNamespace()
    # reuse SongSpace.metrics as an unbound function
    metrics = SongSpace.metrics(
        ss, predictions=preds, labels=labels, classes=["c1", "c2"]
    )
    assert "c1" in metrics and "c2" in metrics
    # c2 should be NaN for metrics because no labeled samples
    assert np.isnan(metrics["c2"]["average_precision"]) and np.isnan(
        metrics["c2"]["roc_auc"]
    )
    # macro metrics present
    assert "macro_average_precision" in metrics and "macro_roc_auc" in metrics


def test_evaluate_raises_on_index_mismatch(tmp_path, monkeypatch):
    ss = make_songspace(tmp_path, monkeypatch)
    # prepare a dataset with two rows and an index
    idx = pd.MultiIndex.from_tuples(
        [("a.wav", 0.0, 1.0), ("b.wav", 0.0, 1.0)],
        names=["file", "start_time", "end_time"],
    )
    df = pd.DataFrame(index=idx, data={"c": [1, 0]})
    ss.datasets["d"] = {"label_df": df}
    # add a classifier stub
    ss.classifiers["clf"] = SimpleNamespace(classes=["c"])

    # monkeypatch predict_on_dataset to return a dataframe with a different index
    def bad_pred(*args, **kwargs):
        return pd.DataFrame(index=pd.Index([0, 1]), data={"c": [0.1, 0.2]})

    monkeypatch.setattr(SongSpace, "predict_on_dataset", bad_pred)
    with pytest.raises(ValueError):
        ss.evaluate("clf", "d")


def test_get_dataset_embeddings_delegates_and_returns_array(tmp_path, monkeypatch):
    ss = make_songspace(tmp_path, monkeypatch)
    # prepare a dataset and monkeypatch _find_matching_window_ids
    idx = pd.MultiIndex.from_tuples(
        [("a.wav", 0.0, 1.0)], names=["file", "start_time", "end_time"]
    )
    df = pd.DataFrame(index=idx, data={"c": [1]})
    ss.datasets["d"] = {"label_df": df}
    monkeypatch.setattr(
        song_space,
        "_find_matching_window_ids",
        lambda db, label_df, project, return_val: [10, 11],
    )
    # ensure the underlying DB returns embeddings
    out = ss.get_dataset_embeddings("d")
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 2


def test__get_unlabeled_samples_warns_when_no_datasets(tmp_path, monkeypatch):
    ss = make_songspace(tmp_path, monkeypatch)
    # ensure no datasets with allow_training=True
    ss.datasets = {}
    with pytest.warns(UserWarning):
        out = ss._get_unlabeled_samples(5, classes=["a", "b"])
    assert out is None


def test_similarity_search_delegates(tmp_path, monkeypatch):
    ss = make_songspace(tmp_path, monkeypatch)
    res = ss.similarity_search(query_samples=["a.wav"], k=3)
    assert res == "SIMILAR"


def test_select_with_classifier_name_calls_select_from_hoplite(tmp_path, monkeypatch):
    ss = make_songspace(tmp_path, monkeypatch)
    # add a classifier object
    model = SimpleNamespace(classes=["a"])
    ss.classifiers["myclf"] = model

    called = {}

    def fake_select_from_hoplite(**kwargs):
        called["classifier"] = kwargs.get("classifier")
        return pd.DataFrame()

    monkeypatch.setattr(song_space, "select_from_hoplite", fake_select_from_hoplite)
    ss.select("myclf", classes=["a"])  # should not raise
    assert "classifier" in called and called["classifier"] is model
