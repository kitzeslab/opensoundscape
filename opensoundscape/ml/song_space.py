from pathlib import Path
import pandas as pd
import warnings
from aru_metadata_parser.parse import ARUFileTimestampParser
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from opensoundscape.ml.shallow_classifier import predict_on_hoplite
from opensoundscape.ml.shallow_classifier import MLPClassifier
from opensoundscape.ml.shallow_classifier import fit_on_hoplite_db
from opensoundscape.ml.loss import BCELossWeakNegatives
from opensoundscape.vector_database import (
    _find_matching_window_ids,
    load_or_create_hoplite_usearch_db,
)

default_datetime_parser = ARUFileTimestampParser()


# utilities for guessing deployment name from file path
def parent_folder_name(file_path):
    """Utility function to extract the parent folder name from a file path"""
    return Path(file_path).parent.name


def two_parents_name(file_path):
    """Utility function to extract "grandparent_parent" folder name from a file path"""
    p = Path(file_path).parent
    gp = p.parent
    return f"{gp.name}_{p.name}"


def second_parent_name(file_path):
    """Utility function to extract the second parent folder name from a file path"""
    return Path(file_path).parent.parent.name


def filename_first_part(file_path):
    """Utility function to extract the part of the filename before the first underscore from a file path"""
    return Path(file_path).stem.split("_")[0]


class SongSpace:
    """SongSpace is a framework for training and applying classifiers, combining a feature extractor and database

    A SongSpace couples a feature extractor (e.g., BirdNET or Perch) with a database that stores embeddings of audio clips
    We can add one or more shallow classifiers, and labeled training and evaluation datasets

    It provides utilities for:
    - fitting a classifier on embeddings with optional validation and early stopping
    - applying a classifier to embeddings in a hoplite database with filtering by metadata and existing scores
    - selecting top-scoring or random clips from the database based on classifier predictions and filters

    The main idea is to enable users to easily complete an active learning loop:
    - start with a few labeled samples and a bunch of unlabeled audio
    - embed everything
    - use similarity search, shallow classifiers, or targeted/random search to find clips
    - review clips and label more data
    - apply the final classifier to select clips for manual verification
    - end with manually verified detections for downstream analysis
    """

    def __init__(self, database, feature_extractor="birdnet"):
        if isinstance(feature_extractor, str):
            _require_bmz()
            import bioacoustics_model_zoo as bmz

            if feature_extractor == "birdnet":
                feature_extractor = bmz.BirdNET()
            elif feature_extractor == "perch":
                feature_extractor = bmz.Perch()
            else:
                raise ValueError(
                    f"Unsupported feature extractor: {feature_extractor}. Supported options are 'birdnet' and 'perch'."
                )
        if isinstance(database, (str, Path)):
            database = load_or_create_hoplite_usearch_db(
                database,
                embedding_dim=feature_extractor.classifier.in_features,
            )
        self.feature_extractor = feature_extractor
        self.database = database
        self.classifiers = {}
        self.datasets = {}
        self.embedding_dim = database.get_embedding_dim()
        self.sample_duration = feature_extractor.preprocessor.sample_duration

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
        return self.database

    def remove_dataset(self, name):
        """Remove a dataset from the SongSpace by name"""
        del self.datasets[name]

    def get_dataset(self, name):
        """return labels_df for dataset name"""
        return self.datasets[name]["label_df"]

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
                Utility functions for common patterns are provided in opensoundscape.ml.song_space, including
                parent_folder_name, two_parents_name, second_parent_name, filename_first_part
                (an LLM would also be great at writing a custom function given your deployment:audio file structure)
            allow_training: if True, allows using this dataset for training classifiers; if False,
                dataset can still be used for validation but not training; default True
            audio_root: if provided, used as prefix for audio files in samples;
                if None, assumes samples already have absolute audio paths
                #TODO: if full paths provided and audio_root provided, convert to relative paths by
                stripping audio_root from the start of the paths in samples before embedding and
                storing in the database
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

        samples_df, _ = _ingest_samples_argument(samples)

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
        all_embedded = []
        for deployment, files in deployment_to_files.items():
            deployment_samples_df = samples_df.loc[files]
            _, sample_info = self.feature_extractor.embed_to_hoplite_db(
                deployment_samples_df,
                db=self.database,
                audio_root=audio_root,
                embedding_exists_mode=embedding_exists_mode,
                deployment=deployment,
                project=dataset_name,
                file_to_datetime=file_to_datetime,
                **kwargs,
                # target_layer=None,
                # wandb_session=None,
                # progress_bar=True,
                # commit_frequency_batches=100,
                # overflow_mode="warn",
                # embedding_dim=None,
                # strict_matching=False,
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
        clf.val_metrics = fit_on_hoplite_db(
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
                f"No datasets with allow_training=True have unlabeled samples for the specified classes. Consider ingesting un-annotated data with ingest_audio() and allow_training=True to get weak negatives."
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
