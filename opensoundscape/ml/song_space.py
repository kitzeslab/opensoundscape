from pathlib import Path
import pandas as pd
import warnings
from aru_metadata_parser.parse import ARUFileTimestampParser
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from opensoundscape.ml.shallow_classifier import (
    predict_on_hoplite,
    MLPClassifier,
    fit_on_hoplite,
)
from opensoundscape.ml.loss import BCELossWeakNegatives
from opensoundscape.vector_database import (
    _find_matching_window_ids,
    load_or_create_hoplite_usearch_db,
)
import json

# helper functions for common patterns of mapping audio files to deployments
# for the file_to_deployment argument in ingest_audio
from opensoundscape.utils import (
    parent_folder_name,
    two_parents_name,
    second_parent_name,
    filename_first_part,
)

default_datetime_parser = ARUFileTimestampParser()


class SongSpace:
    """SongSpace is a framework for training and applying classifiers, combining a feature extractor and database

    A SongSpace couples a feature extractor (e.g., BirdNET or Perch) with a database that stores embeddings of audio clips
    We can add one or more shallow classifiers, and labeled training and evaluation datasets

    It provides utilities for:
    - ingesting audio datasets by saving their deep learning embeddings in a database
    - creating and evaluating (shallow) classifiers
    - applying a classifier to embeddings in a hoplite database with filtering by metadata and scores
    - selecting top-scoring or random clips from the database based on classifier predictions and filters
    - embedding-based similarity search

    The main purpose of this class is to enable users to easily complete an active learning loop:
    - start with a few labeled samples and a bunch of unlabeled audio
    - embed everything
    - use similarity search, shallow classifiers, or targeted/random search to find clips
    - review clips and label more data
    - apply the final classifier to select clips for manual verification
    - end with manually verified detections for downstream analysis
    - potentially repeat with other species/classes
    """

    @classmethod
    def open(cls, path, feature_extractor=None):
        """Open an existing SongSpace from a specified path

        if the feature_extractor is not one of the registered bioacoustics model zoo options
        ("bs-convnext", "birdnet", "perch", "perch2"), create the feature extractor used previously,
        then pass it to this method.
        """
        path = Path(path)
        # load songspace metadata
        with open(path / "songspace.json") as f:
            metadata = json.load(f)
        if metadata["feature_extractor"]["source"] == "bioacoustics_model_zoo":
            # for standard feature extractors, we can just initialize by passing the name
            feature_extractor = metadata["feature_extractor"]["key"]
        elif feature_extractor is None:
            raise ValueError(
                f"""The SongSpace at {path} was created with a custom feature extractor, so you must
                 provide the same feature extractor to open it. The metadata indicates the feature
                 extractor used was {metadata['feature_extractor']}."""
            )
        instance = cls(
            path=path,
            feature_extractor=feature_extractor,
            sample_duration=metadata["sample_duration"],
        )

        # load classifiers
        for name, clf_path in metadata["classifiers"].items():
            instance.classifiers[name] = MLPClassifier.load(path / clf_path)

        # load datasets
        for name, ds_info in metadata["datasets"].items():
            instance.datasets[name] = {
                "allow_training": ds_info["allow_training"],
                "audio_root": ds_info["audio_root"],
                "label_df": pd.read_pickle(path / ds_info["label_df_path"]),
            }
        return instance

    def save(self):
        """Save the SongSpace metadata to the SongSpace path, so that it can be re-loaded later with SongSpace.open()"""
        # TODO: use a hash of feature extractor and preprocessor to ensure same model when re-loading the SongSpace

        # save datasets to pickle in the SongSpace directory
        (Path(self.path) / "datasets").mkdir(exist_ok=True)
        for name, ds in self.datasets.items():
            ds["label_df"].to_pickle(
                Path(self.path) / "datasets" / f"{name}_labels.pkl"
            )
        # save classifiers
        (Path(self.path) / "classifiers").mkdir(exist_ok=True)
        for name, clf in self.classifiers.items():
            clf.save(Path(self.path) / "classifiers" / f"{name}_classifier.mlp")

        # save metadata json for re-loading the SongSpace with the same models and datasets
        metadata = {
            "feature_extractor": self.feature_extractor_info,
            "datasets": {
                name: {
                    "allow_training": ds["allow_training"],
                    "audio_root": ds["audio_root"],
                    "label_df_path": f"datasets/{name}_labels.pkl",
                }
                for name, ds in self.datasets.items()
            },
            "classifiers": {
                name: f"classifiers/{name}_classifier.mlp"
                for name, _ in self.classifiers.items()
            },
            "sample_duration": self.sample_duration,
        }
        with open(Path(self.path) / "songspace.json", "w") as f:
            json.dump(metadata, f)
        print(
            f"Saved SongSpace to {self.path} with {len(self.classifiers)} classifiers and {len(self.datasets)} datasets."
        )

    def __init__(self, path, feature_extractor="perch2", sample_duration=None):
        self.path = path

        # create directory for this SongSpace, which will hold the database and metadata
        Path(path).mkdir(parents=True, exist_ok=True)

        if isinstance(feature_extractor, str):
            _require_bmz()
            import bioacoustics_model_zoo as bmz

            self.feature_extractor_info = {
                "source": "bioacoustics_model_zoo",
                "key": feature_extractor,
            }
            if feature_extractor == "bs-convnext":
                feature_extractor = bmz.BirdSetConvNeXT()
            elif feature_extractor == "birdnet":
                feature_extractor = bmz.BirdNET()
            elif feature_extractor == "perch":
                feature_extractor = bmz.Perch()
            elif feature_extractor == "perch2":
                feature_extractor = bmz.Perch2()
            else:
                raise ValueError(
                    f"Unsupported feature extractor: {feature_extractor}. Supported options are "
                    "'bs-convnext','birdnet', 'perch', and 'perch2' (or pass your own model)."
                )

        else:
            self.feature_extractor_info = {
                "source": "custom",
                "key": feature_extractor.__class__.__name__,
            }
        version = getattr(feature_extractor, "version", "unknown_version")
        self.feature_extractor_info["version"] = version

        database = load_or_create_hoplite_usearch_db(
            path,
            embedding_dim=feature_extractor.classifier.in_features,
        )
        if database.get_embedding_dim() != feature_extractor.classifier.in_features:
            raise ValueError(
                f"Database embedding dimension {database.get_embedding_dim()} does not match feature extractor output dimension {feature_extractor.classifier.in_features}"
            )
        self.feature_extractor = feature_extractor
        self._database = database
        self.classifiers = {}
        self.datasets = {}
        self.embedding_dim = database.get_embedding_dim()
        if sample_duration is None:
            sample_duration = feature_extractor.sample_duration
        self.sample_duration = sample_duration

    def remove_classifier(self, name):
        """Remove a classifier from the SongSpace by name"""
        del self.classifiers[name]

    def list_datasets(self):
        """List the names of the datasets currently in the SongSpace"""
        return list(self.datasets.keys())

    def list_classifiers(self):
        """List the names of the classifiers currently in the SongSpace"""
        return list(self.classifiers.keys())

    @property
    def db(self):
        """alias for self.database"""
        return self.database

    @property
    def database(self):
        """The database object used to store embeddings for this SongSpace

        property to protect from accidental modification
        """
        return self._database

    def remove_dataset(self, name):
        """Remove a dataset from the SongSpace by name"""
        del self.datasets[name]

    def get_dataset(self, name):
        """return labels_df for dataset name"""
        return self.datasets[name]["label_df"]

    def update_dataset_audio_root(self, name, new_audio_root):
        """Update the audio_root for a given dataset, which is used as the prefix for audio file paths when embedding new samples and searching for existing embeddings in the database

        This is useful if you need to move your audio files after ingesting a dataset, or if you originally ingested with incorrect audio paths.

        Note that this does not change the file paths in the label_df, but rather updates the audio_root that is prefixed to those file paths when embedding new samples or searching for existing embeddings in the database.
        """
        self.datasets[name]["audio_root"] = new_audio_root

    def ingest_audio(
        self,
        samples,
        dataset_name,
        file_to_deployment=parent_folder_name,
        allow_training=True,
        audio_root=None,
        embedding_exists_mode="skip",
        file_to_datetime=default_datetime_parser.parse,
        **kwargs,
    ):
        """Embed samples using the feature extractor and store in a new or existing dataset

        Args:
            samples: dataframe with columns "file", "start_time", "end_time" specifying clips to embed
            dataset_name: name of the dataset to store the embeddings in
                - if existing, combines with existing dataset of the same name, taking the new
                  labels in the case of conflicts
                - if not existing, creates a new dataset with the given name, using allow_training
                  and audio_root to set up the dataset parameters
                Also uses dataset_name as the 'project' name for the deployment in the database
            file_to_deployment: str, function, or dictionary mapping filenames to deployment names
                - if function, should take a single argument (filename: str) and return a deployment name (str)
                - if dictionary or pd.Series, should map filenames (str) to deployment names (str)
                - if str, the name of the deployment that all samples will be associated with
                - if deployment does not exist in db, it will be created
                Utility functions for common patterns are provided in opensoundscape.utils, including
                parent_folder_name, two_parents_name, second_parent_name, filename_first_part
                (an LLM would also be great at writing a custom function given your deployment:audio file structure)
            allow_training: if True, allows using this dataset for training classifiers; if False,
                dataset can still be used for validation but not training; default True
            audio_root: if provided, used as prefix for audio files in samples;
                if None, assumes samples already have absolute audio paths
                if full paths provided and audio_root provided, converts to relative paths by
                stripping audio_root from the start of the paths in samples before embedding and
                storing in the database
                (see also: update_dataset_audio_root() to update audio_root if you move the entire audio dataset)
            embedding_exists_mode: 'skip', 'error', or 'add' [default: 'skip']
                how to handle cases where an embedding already exists in the database
                # TODO impement 'replace'
                skip: skip embedding and keep existing embedding
                error: raise an error if an embedding already exists for a clip in samples
                add: add a new embedding alongside the existing one (e.g. for augmentated variations of same clip)
            file_to_datetime: optional function or dictionary mapping filenames to datetime objects
                - used to set recording start times in the database
                Default: uses a flexible parser from aru_metadata_parser.parse handling most formats
            **kwargs: additional keyword arguments to pass to the feature extractor's embed() method

        """
        from opensoundscape.ml.datasets import _ingest_samples_argument

        samples_df, _ = _ingest_samples_argument(
            samples, sample_duration=self.sample_duration
        )

        # if audio_root is provided and the file paths in samples_df are absolute, all starting with audio_root,
        # convert to relative paths by stripping audio_root from the start of the paths in samples_df
        if audio_root is not None:
            if dataset_name in self.datasets:  # ensure it matches existing audio_root
                existing_audio_root = self.datasets[dataset_name]["audio_root"]
                if existing_audio_root != audio_root:
                    raise ValueError(
                        f"""Provided audio_root {audio_root} does not match existing audio_root
                         {existing_audio_root} for dataset {dataset_name}. One dataset cannot have
                         multiple audio roots."""
                    )
            audio_root = str(audio_root)
            # short-circuit if first path doesn't start with audio_root, to avoid unnecessary processing
            # (do nothing if audio_root is not the beginning of all paths)
            if samples_df.iloc[0]["file"].startswith(audio_root) and all(
                samples_df["file"].str.startswith(audio_root)
            ):
                samples_df["file"] = samples_df["file"].str[len(audio_root) :]

        # compute deployment name for each audio file
        unique_files = samples_df.index.get_level_values("file").unique()
        deployment_to_files = {}
        for f in unique_files:
            if isinstance(file_to_deployment, dict):
                d = file_to_deployment[f]
            elif callable(file_to_deployment):
                d = file_to_deployment(f)
            elif isinstance(file_to_deployment, str):
                d = file_to_deployment
            elif file_to_deployment is None:
                d = None
            else:
                raise ValueError(
                    f"Invalid file_to_deployment argument: {file_to_deployment}. Must be a string, a dictionary mapping filenames to deployment names, or a function that takes a filename and returns a deployment name."
                )
            if d in deployment_to_files:
                deployment_to_files[d].append(f)
            else:
                deployment_to_files[d] = [f]

        # loop over each deployment, embedding the corresponding samples
        # all_embedded = []
        for deployment, files in deployment_to_files.items():
            deployment_samples_df = samples_df.loc[files]
            # was trying to keep track of sample_id as embeddings are inserted, but
            # this function skips already-embedded samples, making this tricky. Instead
            # we will look up the sample ids as needed based on (file,start_time,end_time) windows
            _, sample_info = self.feature_extractor.embed_to_hoplite_db(
                deployment_samples_df,
                db=self.database,
                audio_root=audio_root,
                embedding_exists_mode=embedding_exists_mode,
                deployment=deployment,
                project=dataset_name,
                file_to_datetime=file_to_datetime,
                **kwargs,
                # wandb_session=None,
                # progress_bar=True,
                # commit_frequency_batches=100,
                # overflow_mode="warn",
                # strict_matching=False, # do we want True here?
                # **dataloader_kwargs,
            )
        if dataset_name in self.datasets:
            # add embedded samples to existing dataset, following embedding_exists_mode behavior
            ds = self.datasets[dataset_name]
            existing_df = ds["label_df"]
            # combine with existing dataset; behavior depends on embedding_exists_mode
            if embedding_exists_mode == "skip":
                # we skipped existing labels, so only update entries for new samples and keep existing labels for overlapping samples
                # (keep in mind that the embeddings could be in the database but not in our label_df if we re-loaded the db and re-initialized this class)
                ds["label_df"] = existing_df.combine_first(samples_df)
            elif embedding_exists_mode == "add" or embedding_exists_mode == "error":
                # concatenate the new samples with the existing ones, allowing duplicates
                ds["label_df"] = pd.concat(
                    [existing_df, samples_df], ignore_index=False
                )
            else:
                raise ValueError(
                    f"Invalid embedding_exists_mode: {embedding_exists_mode}. Must be one of 'skip', 'error', or 'add'."
                )
        else:  # new dataset
            self.datasets[dataset_name] = {
                "label_df": samples_df,
                "allow_training": allow_training,
                "audio_root": audio_root,
            }

    def similarity_search(
        self,
        query_samples,
        k=5,
        exact_search=False,
        search_subset_size=None,
        target_score=None,
        audio_root=None,
        search_kwargs=None,
        **embedding_kwargs,
    ):
        """Find the k most similar embeddings in the database to each query audio sample

        Args:
            query_samples: audio file path, list of files, or dataframe with columns "file", "start_time", "end_time" specifying clips to embed and search for
            k: number of similar samples to return; default 5
            exact_search: default (False) uses an approximate nearest neighbor search for speed;
                if True, uses exact search for maximum recall but slower speed
            search_subset_size: if provided, limits the search to a random subset of all samples
            target_score: if provided, returns samples close to the target similarity score rather than _most_ similar samples
                - useful for finding samples that are similar but not too similar to the query samples
            audio_root: if provided, used as prefix for audio files in query_samples;
                if None, assumes query_samples already have absolute audio paths
            search_kwargs: dict of additional keyword arguments passed to db.ui.search() or
                brutalism.threaded_brute_search() if exact_search=True
                exact_search=False: radius, threads, exact, log, progress
                exact_search=True: batch_size, max_workers, rng_seed
            **embedding_kwargs: additional keyword arguments passed to self.embed(), such as
                batch_size and num_workers
        Returns:
            A dataframe with the same columns as the database metadata and an additional 'similarity' column, sorted by similarity to the query embedding
        """
        # use the database's built-in similarity search function
        results = self.feature_extractor.similarity_search_hoplite_db(
            query_samples,
            self.database,
            num_results=k,
            exact_search=exact_search,
            search_subset_size=search_subset_size,
            target_score=target_score,
            audio_root=audio_root,
            search_kwargs=search_kwargs,
            **embedding_kwargs,
        )
        return results

    def fit_classifier(
        self,
        classes,
        train_datasets,
        validation_dataset,
        weak_negatives_proportion=2,
        batch_size=128,
        steps=1000,
        optimizer=None,
        criterion=None,
        device="cpu",
        early_stopping_patience=None,
        logging_interval=100,
        validation_interval=1,
        classifier_hidden_layers=(),
        weak_negatives_weight=0.01,
    ):
        """Fit a classifier on embeddings from the database for a given dataset

        Note: Before fitting a classifier, ingest and create audio datasets with ingest_audio()

        Args:
            classes: list of class names to train the classifier for; if None, trains for every class in the dataset(s)
            train_datasets: list of dataset names to use for training; must have been added with ingest_audio()
            validation_dataset: dataset name to use for validation
                if None, skips validation
            weak_negatives_proportion: ratio of weak negatives to positives to add to the training data
                selects random unlabeled samples from the database and treats as no-species samples, but with
                a small weight in the loss function
                default 2 means adding 2 weak negatives for every labeled sample; if 0, does not add any weak negatives
                ignored if criterion is passed
            embedding_batch_size: batch size for embedding; default 1
            embedding_num_workers: number of workers for embedding; default 0
            batch_size, steps, optimizer, criterion, device: model fitting parameters, see fit()
            early_stopping_patience: if provided, training will stop early if validation loss doesn't improve
                for this many steps (not validation evaluations)
                [Default: None, which means no early stopping]
            logging_interval: how often to print training progress; progress is logged every logging_interval steps
                when validation is performed
            validation_interval: how often to validate the model during training; if validation_dataset_name is provided,
                validation is performed every validation_interval steps
            audio_root: if provided, used as prefix for audio files in train_df and validation_df;
                if None, assumes train_df and validation_df already have absolute audio paths
            classifier_hidden_layers: tuple of hidden layer sizes for the MLPClassifier;
                default is () for no hidden layers (i.e. linear probe / logistic regression)
            weak_negatives_weight: weight for the weak negative samples in the loss function
                default 0.01; ignored if criterion is passed
        Returns: new classifier
        """
        # prepare training data by concatenating the label_dfs for the specified datasets
        train_dfs = []
        if classes is None:
            # create class list as union of all classes in the specified datasets
            classes = set()
            for dataset_name in train_datasets:
                label_df = self.get_dataset(dataset_name)
                classes = classes.union(set(label_df.columns))
            if validation_dataset is not None:
                val_label_df = self.get_dataset(validation_dataset)
                classes = classes.union(set(val_label_df.columns))
            classes = sorted(list(classes))  # avoids non-deterministic class order

        # aggregate training label dataframes, filling in missing columns with NaN
        for dataset_name in train_datasets:
            label_df = self.get_dataset(dataset_name)
            # add missing columns for any classes not in this dataset, filling with NaN
            missing_classes = set(classes) - set(label_df.columns)
            for c in missing_classes:
                label_df[c] = np.nan
            # subset and re-order columns to match classes
            train_dfs.append(label_df[classes])
        # add weak negatives to training
        if weak_negatives_proportion > 0:
            n_positives = sum(len(df) for df in train_dfs)
            n_weak_negatives = int(n_positives * weak_negatives_proportion)
            weak_negatives = self._get_unlabeled_samples(
                n_weak_negatives, classes=classes
            )
            if weak_negatives is not None:
                train_dfs.append(weak_negatives[classes])
        train_df = pd.concat(train_dfs)
        validation_df = (
            self.get_dataset(validation_dataset) if validation_dataset else (None, [])
        )
        print(
            f"training classifier for {len(classes)} classes with {len(train_df)} training samples and {len(validation_df) if validation_df is not None else 'no'} validation samples"
        )
        # initialize and fit classifier
        clf = MLPClassifier(
            input_size=self.embedding_dim,
            output_size=len(classes),
            hidden_layer_sizes=classifier_hidden_layers,
            classes=classes,
        )
        if criterion is None:
            criterion = BCELossWeakNegatives(weak_negative_weight=weak_negatives_weight)
        clf.val_metrics = fit_on_hoplite(
            classifier=clf,
            hoplite_db=self.database,
            train_df=train_df,
            validation_df=validation_df,
            batch_size=batch_size,
            steps=steps,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            validation_interval=validation_interval,
            logging_interval=logging_interval,
            early_stopping_patience=early_stopping_patience,
        )
        return clf

    def add_classifier(self, name, model):
        """Add a classifier to the SongSpace with a given name"""
        if name in self.classifiers:
            raise ValueError(
                f"A classifier with name {name} already exists. Please choose a different name "
                "or remove the existing classifier before adding a new one with the same name."
            )
        if not hasattr(model, "classes"):
            raise ValueError(
                f"The model must have a 'classes' attribute that lists the class names."
            )
        self.classifiers[name] = model

    def predict_on_dataset(
        self, classifier_name, dataset_name, batch_size=1024, return_df=True
    ):
        """Apply a classifier to a dataset and return predictions as a dataframe with the same index as the dataset's label_df and columns for each class"""
        if classifier_name not in self.classifiers:
            raise ValueError(
                f"No classifier with name {classifier_name} found in SongSpace"
            )
        if dataset_name not in self.datasets:
            raise ValueError(f"No dataset with name {dataset_name} found in SongSpace")
        classifier = self.classifiers[classifier_name]
        label_df = self.get_dataset(dataset_name)
        preds = predict_on_hoplite(
            db=self.database,
            samples=label_df,
            classifier=classifier,
            clip_duration=self.sample_duration,
            batch_size=batch_size,
            return_df=return_df,
        )
        return preds

    def metrics(self, predictions, labels, classes):
        """Compute evaluation metrics for a set of predictions and true labels"""
        metrics = {}
        for class_name in classes:
            y_true = labels[class_name].values
            y_pred = predictions[class_name].values
            # only compute metrics for samples with labels (i.e. not NaN)
            mask = ~np.isnan(y_true)
            if np.sum(mask) == 0:
                metrics[class_name] = {
                    "average_precision": np.nan,
                    "roc_auc": np.nan,
                }
                continue
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            metrics[class_name] = {
                "average_precision": average_precision_score(y_true, y_pred),
                "roc_auc": roc_auc_score(y_true, y_pred),
            }
        # macro average metrics, ignoring nans
        map = np.nanmean([m["average_precision"] for m in metrics.values()])
        mauroc = np.nanmean([m["roc_auc"] for m in metrics.values()])
        metrics["macro_average_precision"] = map
        metrics["macro_roc_auc"] = mauroc
        return metrics

    def evaluate(self, classifier_name, dataset_name, batch_size=1024):
        """Evaluate a classifier on a specified dataset and return metrics"""
        preds = self.predict_on_dataset(classifier_name, dataset_name, batch_size)
        label_df = self.datasets[dataset_name]["label_df"]
        classes = self.classifiers[classifier_name].classes
        # check that preds and label_df have the same index
        if not preds.index.equals(label_df.index):
            raise ValueError(
                f"Predictions and labels have different indices. Cannot evaluate. Preds index: {preds.index}, label_df index: {label_df.index}"
            )
        # compute metrics for each class and macro averages
        return self.metrics(predictions=preds, labels=label_df, classes=classes)

    def get_dataset_embeddings(self, dataset_name):
        """Utility to get the embeddings and labels for a given dataset as numpy arrays"""
        label_df = self.get_dataset(dataset_name)
        # TODO allow random for return_val?
        window_ids = _find_matching_window_ids(
            self.database, label_df, project=dataset_name, return_val="first"
        )
        embeddings = self.database.get_embeddings_batch(window_ids)
        return embeddings

    def _get_unlabeled_samples(
        self, n_samples, classes, dataset_list=None, check_labels=True
    ):
        """Utility to get a specified number of random unlabeled samples from the database as a dataframe

        we'll define 'unlabeled' as any sample in any dataset marked as allow_training=True that has no labels for the specified classes

        Args:
            n_samples: number of unlabeled samples to return; if more samples are available, a random subset is returned
            classes: list of class names to check for labels; samples with no labels for any of these classes are considered unlabeled
            dataset_list: list of dataset names to consider when looking for unlabeled samples; if None, considers all datasets with allow_training=True
            check_labels: whether to check if samples have labels for the specified classes
                - if False, returns random samples from the specified datasets without checking for labels
                This is faster when selecting from large, unlabeled datasets
        """
        # TODO: should be checking db rather than datasets? db could have many samples not listed in datasets?

        if dataset_list is None:
            # use all datasets with allow_training=True
            dataset_list = list(
                [d for d in self.datasets.keys() if self.datasets[d]["allow_training"]]
            )
        # find all samples in datasets with allow_training=True that have no labels for the specified classes
        unlabeled_dfs = []
        for dataset_name in dataset_list:
            label_df = self.datasets[dataset_name]["label_df"]
            if check_labels:
                # identify samples with no labels for any of the specified classes
                mask = label_df[classes].isna().all(axis=1)
                label_df = label_df[mask]
            # if more than n_samples, subset now to avoid massive concatenation
            unlabeled_dfs.append(label_df.sample(n=min(n_samples, len(label_df))))

        if len(unlabeled_dfs) == 0:
            # welp, we don't have weak negatives. No biggie, just suggest user ingests un-annotated data
            warnings.warn(
                f"""No datasets with allow_training=True have unlabeled samples for the specified
                 classes. Consider ingesting un-annotated data with ingest_audio() and
                 allow_training=True to get weak negatives."""
            )
            return None
        else:
            unlabeled_df = pd.concat(unlabeled_dfs)
        # sample up to n_samples from the unlabeled dataframe
        unlabeled_df = unlabeled_df.sample(n=min(n_samples, len(unlabeled_df)))
        return unlabeled_df


def _require_bmz():
    try:
        import bioacoustics_model_zoo as bmz
    except ImportError:
        raise ImportError(
            "The 'bioacoustics_model_zoo' package is required to use string names for feature extractors in SongSpace. Please install it with 'pip install bioacoustics_model_zoo' or specify a feature extractor object directly when initializing SongSpace."
        )
