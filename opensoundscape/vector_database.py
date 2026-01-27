"""utilities for integrating with HopLite vector database


(possibly other vector database libraries in the future)
"""

from datetime import datetime
import numpy as np
from pathlib import Path
import warnings
import pandas as pd


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
    try:
        import perch_hoplite
        from perch_hoplite.db.sqlite_usearch_impl import (
            SQLiteUSearchDB,
            get_default_usearch_config,
        )
    except ImportError as e:
        raise ImportError(
            "hoplite package is required to use hoplite databases. "
            "Please install with `pip install perch-hoplite`"
        ) from e

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
    # insert the embeddings one-by-one to the hoplite db
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
            db.insert_window(
                recording_id=recording_id,
                embedding=batch_embeddings[j],
                offsets=[start_time, end_time],
            )
        except RuntimeError as e:
            # duplicate key error, window already exists
            if 'Duplicate key' in str(e):
                failed_to_insert.append(
                    (file, start_time, end_time, "duplicate window")
                )
            else:
                raise e
    return failed_to_insert


def _handle_existing_windows(
    db,
    clips,
    embedding_exists_mode,
    deployment_id=None,
    deployment_name=None,
    project=None,
    rounding_precision=3,
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

    from ml_collections import config_dict

    if embedding_exists_mode in ["skip", "error"]:

        # first make list of files in input clips dataframe
        file_list = (
            clips.index.get_level_values(0).to_series().astype(str).unique().tolist()
        )
        file_list = [str(Path(f)) for f in file_list]

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

        # get window ids from db matching these recordings (and deployment/project if specified)
        window_ids = db.match_window_ids(
            recordings_filter=recordings_filter, deployments_filter=deployments_filter
        )
        windows = db.get_all_windows(
            filter=config_dict.create(isin=dict(id=window_ids))
        )

        def resolve_path(rec):
            p = rec.filename
            if isinstance(clips.index.levels[0][0], Path):
                p = Path(p)
            else:
                p = str(p)
            return p

        # map of existing recording ids to resolved paths
        id_to_recording = {rec.id: resolve_path(rec) for rec in db.get_all_recordings()}
        # be ware of type mismatch with Path vs str and float vs float32 or np.float64
        existing_index_tuples = {
            (
                id_to_recording[w.recording_id],
                w.offsets[0],
                w.offsets[1],
            )
            for w in windows
        }

        # MultiIndex into an Index of tuples
        flat_clips = clips.index.to_flat_index()
        flat_existing = pd.Index(existing_index_tuples)

        # Normalize the tuples (Vectorized rounding of temporal values)
        def normalize_flat(idx):
            return pd.Index(
                [
                    (
                        # ignore non-relevant Path differences like // vs /
                        str(Path(t[0])),
                        round(float(t[1]), rounding_precision),
                        round(float(t[2]), rounding_precision),
                    )
                    for t in idx
                ]
            )

        norm_clips = normalize_flat(flat_clips)
        norm_existing = normalize_flat(flat_existing)

        # filter df to clips without existing embeddings
        diff_idx = norm_clips.difference(norm_existing)

        if embedding_exists_mode == "error" and len(diff_idx) > 0:
            raise ValueError(
                'Some embeddings exist in db and embedding_exists_mode="error"'
            )

        # subset the clip dataframe inplace to exclude overlapping_idxs
        mask = norm_clips.isin(diff_idx)
        clips.drop(clips.index[~mask], inplace=True)

        if len(clips) == 0:
            # all samples already have embeddings, nothing to do
            print("all samples already have embeddings in the database")
            return db, {}
        else:
            print(f"embedding {len(clips)} new windows to database")
    # else: embedding_exists_mode == "add", add more embeddings even if matching
    # existing windows -> no subsetting needed


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
