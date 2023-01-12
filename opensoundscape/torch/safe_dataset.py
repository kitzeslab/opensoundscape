"""Dataset wrapper to handle errors gracefully in Preprocessor classes

A SafeDataset handles errors in a potentially misleading way: If an error is
raised while trying to load a sample, the SafeDataset will instead load a
different sample. The indices of any samples that failed to load will be
stored in ._invalid_indices.

The behavior may be desireable for training a model, but could cause silent
errors when predicting a model (replacing a bad file with a different file),
and you should always be careful to check for ._invalid_indices after using
a SafeDataset.

based on an implementation by @msamogh in nonechucks
(github.com/msamogh/nonechucks/)
"""
import warnings


class SafeDataset:
    """A wrapper for a Dataset that handles errors when loading samples

    WARNING: When iterating, will skip the failed sample, but when using within
    a DataLoader, finds the next good sample and uses it for
    the current index (see __getitem__).

    Note that this class does not subclass DataSet. Instead, it contains a
    `.dataset` attribute that is a DataSet (or a Preprocessor, which subclasses
    DataSet).

    Args:
        dataset: a torch Dataset instance or child such as a Preprocessor
        eager_eval: If True, checks if every file is able to be loaded during
            initialization (logs _valid_indices and _invalid_indices)

    Attributes: _vlid_indices and _invalid_indices can be accessed later to check
    which samples raised Exceptions. _invalid_samples is a set of all index values
    for samples that raised Exceptions.

    Methods:
        _build_index():
            tries to load each sample, logs _valid_indices and _invalid_indices
        __getitem__(index):
            If loading an index fails, keeps trying the next index until success
        _safe_get_item():
            Tries to load a sample, returns None if error occurs
    """

    def __init__(self, dataset, invalid_sample_behavior, eager_eval=False):
        """Creates a `SafeDataset` wrapper on a DataSet to handle bad samples

        Args:
            dataset: a Pytorch DataSet object
            eager_eval=False: try to load all samples when object is created
            invalid_sample_behavior: what to do when loading a sample results in an error
                - "substitute": pick another sample to load
                - "raise": raise the error
                - "none": return None

        Returns:
            `SafeDataset` wrapper around `dataset`
        """

        self.dataset = dataset
        self.invalid_sample_behavior = invalid_sample_behavior
        self.eager_eval = eager_eval
        # These will contain indices over the original dataset. The indices of
        # the safe samples will go into _valid_indices and similarly for invalid
        # samples in _invalid_indices. _invalid_samples holds the actual names
        self._valid_indices = []
        self._invalid_indices = []
        self._invalid_samples = set()

        # If eager_eval is True, we build the full index of valid/invalid samples
        # by attempting to access every sample in self.dataset.
        if self.eager_eval is True:
            self._build_index()

    def _safe_get_item(self, idx):
        """Returns None instead of throwing an error when dealing with an
        invalid sample, and also builds an index of valid and invalid samples as
        and when they get accessed.
        """
        invalid_idx = False
        try:
            # differentiates IndexError occuring here from one occuring during
            # sample loading
            if idx >= len(self.dataset):
                invalid_idx = True
                raise IndexError("index exceeded end of self.dataset")
            sample = self.dataset[idx]
            if idx not in self._valid_indices:
                self._valid_indices.append(idx)
            return sample
        except Exception as exc:
            if isinstance(exc, IndexError) and invalid_idx:
                raise
            if idx not in self._invalid_indices:
                self._invalid_indices.append(idx)
            # store the actual sample names also?
            sample = self.dataset.label_df.index[idx]
            if isinstance(sample, tuple):
                # just get file path, discard start/end time #TODO revisit choice
                sample = sample[0]
            if sample not in self._invalid_samples:
                self._invalid_samples.add(sample)

            return None

    def _build_index(self):
        """load every sample to determine if each is valid"""
        for idx in range(len(self.dataset)):
            # The returned sample is deliberately discarded because
            # self._safe_get_item(idx) is called only to classify every index
            # into either _valid_indices or _invalid_indices.
            _ = self._safe_get_item(idx)

    def _reset_index(self):
        """Resets the valid and invalid samples indices, & invalid sample list."""
        self._valid_indices = []
        self._invalid_indices = []
        self._invalid_samples = set()

    def report(self, log=None):
        """write _invalid_samples to log file, give warning, & return _invalid_samples"""
        if len(self._invalid_samples) > 0:
            msg = (
                f"There were {len(self._invalid_samples)} "
                "sample(s) that raised errors and were skipped."
            )
            if log is not None:
                with open(log, "w") as f:
                    for p in self._invalid_samples:
                        f.write(p + "\n")
                msg += f"The invalid file paths are logged in {log}"
            warnings.warn(msg)
        return self._invalid_samples

    @property
    def is_index_built(self):
        """Returns True if all indices of the original dataset have been
        classified into _valid_indices or _invalid_indices.
        """
        return len(self.dataset) == len(self._valid_indices) + len(
            self._invalid_indices
        )

    @property
    def num_samples_examined(self):
        """count of _valid_indices + _invalid_indices"""
        return len(self._valid_indices) + len(self._invalid_indices)

    def __len__(self):
        """Returns the length of the original dataset.
        NOTE: This is different from the number of actually valid samples.
        """
        return len(self.dataset)

    def __iter__(self):
        return (
            self._safe_get_item(i)
            for i in range(len(self))
            if self._safe_get_item(i) is not None
        )

    def __getitem__(self, idx):
        """If loading an index fails, behavior depends on self.invalid_sample_behavior

        self.invalid_sample_behavior = {
            "substitute": pick another sample,
            "raise": raise the error
            "none": return None
        """
        if self.invalid_sample_behavior == "substitute":
            attempts = 0
            while attempts < len(self.dataset):
                sample = self._safe_get_item(idx)
                if sample is not None:
                    return sample
                idx += 1
                attempts += 1
                idx = idx % len(self.dataset)  # loop around end to beginning
            raise IndexError(
                "None of the samples in the SafeDataset loaded. "
                "All samples caused exceptions during preprocessing. "
            )
        elif (
            self.invalid_sample_behavior == "raise"
            or self.invalid_sample_behavior == "none"
        ):
            try:
                sample = self.dataset[idx]
                if idx not in self._valid_indices:
                    self._valid_indices.append(idx)
                return sample
            except Exception:
                if idx not in self._invalid_indices:
                    self._invalid_indices.append(idx)
                if self.invalid_sample_behavior == "none":
                    return None
                else:  # raise the Exception
                    raise
        else:
            raise ValueError(
                f"invalid_sample_behavior must be 'substitute','raise', or 'none'. "
                f"Got {self.invalid_sample_behavior}"
            )
