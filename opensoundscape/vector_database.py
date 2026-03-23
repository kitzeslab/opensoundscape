"""utilities for integrating with HopLite vector database


(possibly other vector database libraries in the future)
"""

from datetime import datetime
import numpy as np
from pathlib import Path
import warnings
import pandas as pd
from collections import defaultdict


def load_or_create_hoplite_usearch_db(db, embedding_dim=None, cfg=None):
    """helper function to load or create a hoplite database object

    Args:
        db: a hoplite database object or a path to a hoplite database (folder)
            - if a path is provided, the database will be created if it does not exist,
                and passing embedding_dim is required in this case
                if it does exist, it will be loaded
            - if a hoplite database object is provided, it will be returned as is
        embedding_dim: int, dimension of the embeddings to be stored in the database
            - only required when creating a new database
        cfg: optional config_dict.ConfigDict object with usearch configuration
            - only used when creating a new database
            - if None, default usearch config will be used
            Keys: 'embedding_dim', 'dtype', 'metric_name', 'expansion_add', 'expansion_search'
            Example (default values):
            ```python
            usearch_cfg = config_dict.ConfigDict()
            usearch_cfg.embedding_dim = embedding_dim
            usearch_cfg.dtype = 'float16'
            usearch_cfg.metric_name = 'IP'
            usearch_cfg.expansion_add = 256
            usearch_cfg.expansion_search = 128
            ```
    Returns:
        a hoplite database object
    """
    # TODO: support in-memory backends in addition to sqlite
    _require_hoplite()
    import perch_hoplite
    from perch_hoplite.db.sqlite_usearch_impl import (
        SQLiteUSearchDB,
        get_default_usearch_config,
    )

    if isinstance(db, (str, Path)):
        db_path = Path(db)
        if db_path.exists():
            print(f"Connecting to existing db at {db_path}")
            db = SQLiteUSearchDB.create(db_path)
            n_recordings = len(db.get_all_recordings())
            print(
                f"Connected database has {db.count_embeddings():,} embeddings from {n_recordings} file{'' if n_recordings == 1 else 's'}."
            )
        else:
            assert (
                embedding_dim is not None
            ), "embedding_dim must be provided when creating a new hoplite database"
            print(f"Creating new db at {db_path}")
            if cfg is None:
                cfg = get_default_usearch_config(embedding_dim)
            db = SQLiteUSearchDB.create(db_path, cfg)
    else:
        assert isinstance(
            db, perch_hoplite.db.interface.HopliteDBInterface
        ), "db must be a hoplite database object or a path to a hoplite database"
    return db


def _insert_embeddings(
    db,
    batch_samples,
    batch_embeddings,
    overflow_mode,
    file_to_id,
    file_to_datetime=None,
    audio_root=None,
    deployment_id=None,
):
    """insert a batch of embeddings into a hoplite database
    Args:
        db: a hoplite database object
        batch_samples: list of AudioSample objects corresponding to the embeddings
        batch_embeddings: np.ndarray of shape (batch_size, embedding_dim)
        overflow_mode: str, one of ["warn", "error", "clip"], how to handle float overflow
            when casting embeddings to database dtype (typically float16)
        file_to_id: dict mapping filename to recording_id in the hoplite db
        file_to_datetime: optional dict or mapping filename to datetime, or function that takes
            filename and returns datetime; if provided, used to set the datetime field when inserting
            new recordings into the hoplite db; if None, datetime will be left as None
        audio_root: optional Path, root directory for audio files, used to store relative paths
            in the database
        deployment_id: optional int, deployment_id to associate with the recordings

    Effects:
        inserts the embeddings into the hoplite database
        adds new recordings to the database as needed
    """
    # check database types and float range
    db_dtype = db.get_embedding_dtype()
    max_float = np.finfo(np.dtype(db_dtype)).max

    # we clip values to the database dtype range before casting, to avoid overfloat -> inf values
    if np.abs(batch_embeddings).max() > max_float:
        if overflow_mode == "warn":
            warnings.warn(
                f"clipping embedding values to database dtype ({db_dtype}) range"
            )
        elif overflow_mode == "error":
            raise ValueError(f"Embeddings exceeded database dtype  ({db_dtype}) range")
        # otherwise clip without warnings/errors
    batch_embeddings = batch_embeddings.clip(-max_float, max_float).astype(
        np.dtype(db_dtype)
    )

    failed_to_insert = []
    window_ids = []  # track and return window_id in database for each item

    # insert each embedding in the batch into the database, one-by-one
    for j, audiosample in enumerate(batch_samples):
        file = audiosample.source
        if audio_root is not None:
            # use the relative path for the source name stored in the db
            file = str(Path(file).relative_to(audio_root))
        if file in file_to_id:  # already in recording table of db
            recording_id = file_to_id[file]
        else:  # add this file to recording table in db
            # file_to_datetime is optional mapping of filename -> datetime
            # or function that takes filename and returns datetime
            if callable(file_to_datetime):
                datetime = file_to_datetime(file)
            elif file_to_datetime and file in file_to_datetime:
                datetime = file_to_datetime[file]
            else:
                datetime = None
            recording_id = db.insert_recording(
                filename=file,
                deployment_id=deployment_id,
                datetime=datetime,
            )
            file_to_id[file] = recording_id
        start_time = audiosample.start_time
        end_time = start_time + audiosample.duration
        try:
            # TODO: use insert_windows_batch
            # TODO: use hoplite's new options for 'raise', 'overwrite', 'ignore', 'add' duplicate keys
            window_id = db.insert_window(
                recording_id=recording_id,
                embedding=batch_embeddings[j],
                offsets=[start_time, end_time],
            )
            window_ids.append(window_id)
        except RuntimeError as e:
            window_ids.append(None)  # placeholder for failed insert
            if "Duplicate key" in str(e):
                # duplicate key error, window already exists
                failed_to_insert.append(
                    (file, start_time, end_time, "duplicate window")
                )
            else:
                raise e
    return window_ids, failed_to_insert


def normalize_index_to_tuples(idx, rounding_precision=6):
    """normalize an index of (filename, start_time, end_time) tuples to account for potential float precision issues and Path vs str differences"""
    if isinstance(idx, (list, np.ndarray)):
        idx = pd.Index(idx)
    if isinstance(idx, pd.MultiIndex):
        idx = idx.to_flat_index()
    return [
        (
            # ignore non-relevant Path differences like // vs /
            str(Path(t[0])),
            # round numeric start/end times
            round(float(t[1]), rounding_precision),
            round(float(t[2]), rounding_precision),
        )
        for t in idx
    ]


def normalize_windows_to_tuples(windows, rounding_precision=6):
    """helper function to convert list of hoplite windows to list of (filename, start_time, end_time) tuples"""
    return [
        (
            str(Path(w.filename)),
            round(float(w.offsets[0]), 6),
            round(float(w.offsets[1]), 6),
        )
        for w in windows
    ]


def get_existing_windows(
    db, files, deployment_id=None, deployment_name=None, project=None
):
    """retrieve db windows for list of files, filtering by deployment/project"""
    _require_hoplite()
    from ml_collections import config_dict

    # normalize paths, this will remove artifacts like double slashes that could cause mismatches with db entries
    file_list = [str(Path(f)) for f in files]

    # establish filters for windows from matching files, optionally constraining to deployment/project
    recordings_filter = config_dict.create(isin=dict(filename=file_list))
    # deployment_id takes precedence over deployment_name
    if deployment_id is not None:
        if project is not None:
            # get deployment ids matching both deployment_id and project
            deployments_filter = config_dict.create(
                eq=dict(id=deployment_id, project=project),
            )
        else:
            deployments_filter = config_dict.create(
                eq=dict(id=deployment_id),
            )
    elif deployment_name is not None:
        if project is not None:
            # get deployment ids matching both deployment_name and project
            deployments_filter = config_dict.create(
                eq=dict(name=deployment_name, project=project),
            )
        else:
            deployments_filter = config_dict.create(
                eq=dict(name=deployment_name),
            )
    else:
        deployments_filter = None

    # map of existing recording ids to resolved paths
    id_to_recording = {rec.id: str(rec.filename) for rec in db.get_all_recordings()}

    # get window ids from db matching these recordings (and deployment/project if specified)
    window_ids = db.match_window_ids(
        recordings_filter=recordings_filter, deployments_filter=deployments_filter
    )
    # returned windows not guaranteed to be in the same order as the window_ids!
    windows = db.get_all_windows(filter=config_dict.create(isin=dict(id=window_ids)))

    # add filename to each window based on recording_id
    for w in windows:
        w.filename = id_to_recording[w.recording_id]

    return windows


def build_index(B):
    index = defaultdict(list)
    for i, t in enumerate(B):
        index[t].append(i)
    return index


def find_matches(A, B):
    index = build_index(B)
    return [tuple(index.get(a, ())) for a in A]


def _find_matching_window_ids(
    db,
    clips,
    deployment_id=None,
    deployment_name=None,
    project=None,
    rounding_precision=6,
    return_val="all",  # all or first
):
    """find window ids in the db matching each of the given clips (defined by their filename, start_time, and end_time), optionally constrained to a deployment/project

    Args:
        db: a hoplite database object
        clips: pd.DataFrame with MultiIndex (filename, start_time, end_time)
        deployment_id: optional int, deployment id to constrain matching
            - if both deployment_id and deployment_name are provided, deployment_id takes precedence
        deployment_name: optional str, deployment name to constrain matching
        project: optional str, project name to constrain matching
        rounding_precision: int, number of decimal places to round start_time and end_time to when matching, to account for potential float precision issues
    Returns:
        list of window_id tuples for each row in clips; eg [(), (0,), (3,1)]
        or list of integer if return_val == "first"
    """
    if len(clips) == 0:
        print("Zero samples passed to _find_matching_window_ids")
        return []

    # first make list of files in input clips dataframe
    file_list = clips.index.get_level_values(0).unique()

    windows = get_existing_windows(
        db,
        file_list,
        deployment_id=deployment_id,
        deployment_name=deployment_name,
        project=project,
    )

    # normalize index tuples to account for potential float precision issues and Path vs str differences
    clip_df_tuples = normalize_index_to_tuples(
        clips.index, rounding_precision=rounding_precision
    )
    db_window_tuples = normalize_windows_to_tuples(
        windows, rounding_precision=rounding_precision
    )

    # find matching window ids for each clip in the input dataframe efficiently
    match_positions = find_matches(clip_df_tuples, db_window_tuples)
    window_ids = [w.id for w in windows]
    if return_val == "first":
        # return just the first matching window id for each clip, or None if no matches
        matching_window_ids = [
            index[0] if len(index) > 0 else None for index in match_positions
        ]
        return matching_window_ids
    return [
        tuple(window_ids[pos] for pos in positions) for positions in match_positions
    ]


def _handle_existing_windows(
    db,
    clips,
    embedding_exists_mode,
    deployment_id=None,
    deployment_name=None,
    project=None,
    rounding_precision=6,
):
    """remove samples from clips dataframe that already have embeddings in the db

    optionally pass project and deployment_id/deployment_name to only match existing windows for
    that deployment/project

    Args:
        db: a hoplite database object
        clips: pd.DataFrame with MultiIndex (filename, start_time, end_time)
        embedding_exists_mode: str, behavior when an embedding already exists for a given
            window. Options are:
                "skip": skip inserting the embedding (default)
                "error": raise an error
                "add": add a new embedding entry to the db with the same source info
        deployment_id: optional int, deployment id to constrain existing window matching
            - if both deployment_id and deployment_name are provided, deployment_id takes precedence
        deployment_name: optional str, deployment name to constrain existing window matching
        project: optional str, project name to constrain existing window matching

    Effects:
        may modify clips dataframe in place to remove samples that already have embeddings in the db

    Returns: None
    """
    if len(clips) == 0:
        # all samples already have embeddings, nothing to do
        print("Zero samples passed to _handle_existing_windows")
        return

    if embedding_exists_mode in ["skip", "error"]:
        # look up all windows in the db that are from the files in the input clips dataframe
        # optionally filtering to deployment/project
        file_list = clips.index.get_level_values(0).unique()
        windows = get_existing_windows(
            db,
            file_list,
            deployment_id=deployment_id,
            deployment_name=deployment_name,
            project=project,
        )

        # normalize index tuples to account for potential float precision issues and Path vs str differences
        clip_df_norm = pd.Index(
            normalize_index_to_tuples(
                clips.index, rounding_precision=rounding_precision
            )
        )
        db_windows_norm = pd.Index(
            normalize_windows_to_tuples(windows, rounding_precision=rounding_precision)
        )

        # filter df to clips without existing embeddings
        diff_idx = clip_df_norm.difference(db_windows_norm)

        if embedding_exists_mode == "error" and len(diff_idx) > 0:
            raise ValueError(
                'Some embeddings exist in db and embedding_exists_mode="error"'
            )

        # subset the clip dataframe in place to exclude overlapping_idxs
        mask = clip_df_norm.isin(diff_idx)
        clips.drop(clips.index[~mask], inplace=True)

    # else: embedding_exists_mode == "add", add more embeddings even if matching
    # existing windows -> no subsetting needed
    # TODO: we probably want to know the window id of existing embeddings


def _collate_search_results(db, results):
    # we have a TopKSearchResults object with .search_results containing
    # a list of num_results SearchResult objects with .window_id and .sort_score

    # extract relevant info for each match into dictionaries
    results_list = []
    for match in results.search_results:
        window = db.get_window(int(match.window_id))
        results_list.append(
            {
                "window_id": int(match.window_id),
                "sort_score": match.sort_score,
                "start_time": window.offsets[0],
                "end_time": window.offsets[1],
                "file": db.get_recording(window.recording_id).filename,
            }
        )
    return results_list


def remove_duplicate_windows(db):
    """utility function to remove duplicate windows from a hoplite database

    duplicate = same (recording, start_time, end_time)

    Args:
        db: a hoplite database object
    """
    windows = db.get_all_windows(include_embedding=False)

    seen = set()
    duplicate_window_ids = []
    for window in windows:
        rec = db.get_recording(window.recording_id)
        key = (rec.filename, window.offsets[0], window.offsets[1])
        if key in seen:
            duplicate_window_ids.append(window.id)
        else:
            seen.add(key)

    for window_id in duplicate_window_ids:
        db.remove_window(window_id)

    print(
        f"Removed {len(duplicate_window_ids)} duplicate windows from database. Now contains "
        f"{db.count_embeddings()} embeddings."
    )


def _check_or_set_model_id(db, model_id):
    """Check that the model ID in the db matches the expected model ID, or set it if not present

    This is a safety check to prevent accidentally using a database with a model that doesn't match
    the one specified by `model_id`. If the db does not have a model ID in its metadata, this function
    will set the model ID to the provided `model_id`. If the db already has a model ID, this function
    will check that it matches the provided `model_id` and raise an error if it does not.

    Args:
        db: Hoplite database object
        model_id: string identifier for the model (e.g. "HawkEars_v1.0.2")
    """
    from ml_collections import ConfigDict

    metadata = db.get_metadata(None)
    if "model" in metadata and "model_id" in metadata["model"]:
        if metadata["model"].get("model_id") != model_id:
            raise ValueError(
                f"Model ID mismatch: database has model ID {metadata['model'].get('model_id')}, but expected {model_id}. "
                f"Please use a database with the correct model ID or update the database's model ID to match."
            )
    else:
        # update the configdict in the "model" metadata field if it exists, otherwise create it
        if "model" in metadata:
            model_metadata = ConfigDict(metadata["model"])
            model_metadata.model_id = model_id
            db.insert_metadata("model", model_metadata)
        else:
            db.insert_metadata("model", ConfigDict({"model_id": model_id}))


def _check_or_set_sample_duration(db, sample_duration):
    """Check that the sample duration in the db matches the expected sample duration, or set it if not present

    This is a safety check to prevent accidentally using a database with a sample duration that doesn't match
    the one specified by `sample_duration`. If the db does not have a sample duration in its metadata, this function
    will set the sample duration to the provided `sample_duration`. If the db already has a sample duration, this function
    will check that it matches the provided `sample_duration` and raise an error if it does not.

    Args:
        db: Hoplite database object
        sample_duration: float, expected sample duration in seconds (e.g. 5.0)
    """
    from ml_collections import ConfigDict

    metadata = db.get_metadata(None)
    if "model" in metadata and "sample_duration" in metadata["model"]:
        if metadata["model"].get("sample_duration") != sample_duration:
            raise ValueError(
                f"Sample duration mismatch: database has sample duration {metadata['model'].get('sample_duration')}, but expected {sample_duration}. "
                f"Please use a database with the correct sample duration or update the database's sample duration to match."
            )
    else:
        # update the configdict in the "model" metadata field if it exists, otherwise create it
        if "model" in metadata:
            model_metadata = ConfigDict(metadata["model"])
            model_metadata.sample_duration = sample_duration
            db.insert_metadata("model", model_metadata)
        else:
            db.insert_metadata(
                "model", ConfigDict({"sample_duration": sample_duration})
            )


def _require_hoplite():
    try:
        import perch_hoplite
        from ml_collections import config_dict
    except ImportError as e:
        raise ImportError(
            "hoplite package is required to use hoplite databases. "
            "Please install with `pip install opensoundscape[hoplite]` to use this functionality."
        ) from e
