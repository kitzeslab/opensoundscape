"""utilities for integrating with HopLite vector database


(possibly other vector database libraries in the future)
"""

import numpy as np
from pathlib import Path
import warnings


def load_or_create_hoplite_usearch_db(db, embedding_dim=None):
    """helper function to load or create a hoplite database object

    Args:
        db: a hoplite database object or a path to a hoplite database (folder)
            - if a path is provided, the database will be created if it does not exist,
                and passing embedding_dim is required in this case
                if it does exist, it will be loaded
            - if a hoplite database object is provided, it will be returned as is
        embedding_dim: int, dimension of the embeddings to be stored in the database
            - only required when creating a new database
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
            usearch_cfg = get_default_usearch_config(embedding_dim)
            db = SQLiteUSearchDB.create(db_path, usearch_cfg)
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
    audio_root=None,
    deployment_id=None,
):
    # insert the embeddings one-by-one to the hoplite db
    # TODO: get the dtype of the embeddings from hoplite db
    max_float16 = np.finfo(np.float16).max

    # we clip values to the float16 range before casting, to avoid overfloat -> inf values
    if np.abs(batch_embeddings).max() > max_float16:
        if overflow_mode == "warn":
            warnings.warn("clipping embedding values to float16 range")
        elif overflow_mode == "error":
            raise ValueError("Embeddings exceeded float16 range")
        # otherwise clip without warnings/errors
    batch_embeddings = batch_embeddings.clip(-max_float16, max_float16).astype(
        np.float16
    )

    # insert each embedding in the batch into the database, one-by-one
    for j, audiosample in enumerate(batch_samples):
        file = audiosample.source
        if audio_root is not None:
            # use the relative path for the source name stored in the db
            file = str(Path(file).relative_to(audio_root))
        if file in file_to_id:  # already in recording table of db
            recording_id = file_to_id[file]
        else:  # add this file to recording table in db
            recording_id = db.insert_recording(
                filename=file, deployment_id=deployment_id
            )
            file_to_id[file] = recording_id
        start_time = audiosample.start_time
        end_time = start_time + audiosample.duration

        db.insert_window(
            recording_id=recording_id,
            embedding=batch_embeddings[j],
            offsets=np.array([start_time, end_time]),
        )


def _handle_existing_windows(db, clips, embedding_exists_mode, audio_root=None):
    """remove samples from clips dataframe that already have embeddings in the db"""

    # TODO: consider wither we should also match on deployment_id and/or project_id
    # as well as (filename, start_time, end_time)
    if embedding_exists_mode in ["skip", "error"]:

        # match dtype of index level

        windows = db.get_all_windows(include_embedding=False)

        def resolve_path(rec_id):
            rec = db.get_recording(rec_id)
            if audio_root is not None:
                p = Path(rec.filename).relative_to(audio_root)
            else:
                p = rec.filename
            if isinstance(clips.index.levels[0][0], Path):
                p = Path(p)
            else:
                p = str(p)
            return p

        id_to_recording = {
            rec.id: resolve_path(rec.id) for rec in db.get_all_recordings()
        }
        # be ware of type mismatch with Path vs str and float vs float32
        existing_index_tuples = {
            (id_to_recording[w.recording_id], float(w.offsets[0]), float(w.offsets[1]))
            for w in windows
        }
        # dataloader.dataset.dataset.label_df
        overlapping_idxs = clips.index.intersection(existing_index_tuples)

        if embedding_exists_mode == "error" and len(overlapping_idxs) > 0:
            raise ValueError(
                'Some embeddings exist in db and embedding_exists_mode="error"'
            )

        # subset the clip dataframe inplace to exclude overlapping_idxs
        clips.drop(overlapping_idxs, inplace=True)

        if len(clips) == 0:
            # all samples already have embeddings, nothing to do
            print("all samples already have embeddings in the database")
            return db, {}
        else:
            print(f"embedding {len(clips)} new windows to database")
    # else: embedding_exists_mode == "add", add more embeddings even if matching
    # existing windows


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
