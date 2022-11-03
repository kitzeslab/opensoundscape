"""Dataset wrapper to handle errors gracefully in Preprocessor classes

A SafeDataset handles errors in a potentially misleading way: If an error is
raised while trying to load a sample, the SafeDataset will instead load a
different sample. The indices of any samples that failed to load will be
stored in ._unsafe_indices.

The behavior may be desireable for training a model, but could cause silent
errors when predicting a model (replacing a bad file with a different file),
and you should always be careful to check for ._unsafe_indices after using
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
            initialization (logs _safe_indices and _unsafe_indices)

    Attributes: _safe_indices and _unsafe_indices can be accessed later to check
    which samples threw errors.

    Methods:
        _build_index():
            tries to load each sample, logs _safe_indices and _unsafe_indices
        __getitem__(index):
            If loading an index fails, keeps trying the next index until success
        _safe_get_item():
            Tries to load a sample, returns tuple of (None, exception) if exception occurs
    """

    def __init__(self, dataset, unsafe_behavior, eager_eval=False):
        """Creates a `SafeDataset` wrapper on a DataSet to handle bad samples

        Args:
            dataset: a Pytorch DataSet object
            eager_eval=False: try to load all samples when object is created
            unsafe_behavior: what to do when loading a sample results in an error
                - "substitute": pick another sample to load
                - "raise": raise the error
                - "none": return None

        Returns:
            `SafeDataset` wrapper around `dataset`
        """

        self.dataset = dataset
        self.unsafe_behavior = unsafe_behavior
        self.eager_eval = eager_eval
        # These will contain indices over the original dataset. The indices of
        # the safe samples will go into _safe_indices and similarly for unsafe
        # samples in _unsafe_indices. _unsafe_samples holds the actual names
        self._safe_indices = []
        self._unsafe_indices = []
        self._unsafe_samples = []

        # If eager_eval is True, we build the full index of safe/unsafe samples
        # by attempting to access every sample in self.dataset.
        if self.eager_eval is True:
            self._build_index()

    def _safe_get_item(self, idx):
        """Returns tuple of None, exception instead of throwing the error when dealing with an
        unsafe sample, and also builds an index of safe and unsafe samples as
        and when they get accessed.
        """
        try:
            # differentiates IndexError occuring here from one occuring during
            # sample loading
            invalid_idx = False
            if idx >= len(self.dataset):
                invalid_idx = True
                raise IndexError("index exceeded end of self.dataset")
            sample = self.dataset[idx]
            if idx not in self._safe_indices:
                self._safe_indices.append(idx)
            return (sample, None)
        except Exception as e:
            if isinstance(e, IndexError) and invalid_idx:
                raise
            if idx not in self._unsafe_indices:
                self._unsafe_indices.append(idx)
            # store the actual sample names also?
            sample = self.dataset.label_df.index[idx]
            if sample not in self._unsafe_samples:
                self._unsafe_samples.append(sample)

            return (None, e)

    def _build_index(self):
        """load every sample to determine if each is safe"""
        for idx in range(len(self.dataset)):
            # The returned sample is deliberately discarded because
            # self._safe_get_item(idx) is called only to classify every index
            # into either safe_samples_indices or _unsafe_samples_indices.
            _ = self._safe_get_item(idx)

    def _reset_index(self):
        """Resets the safe and unsafe samples indices, & unsafe sample list."""
        self._safe_indices = self._unsafe_indices = self._unsafe_samples = []

    def report(self, log=None):
        if len(self._unsafe_samples) > 0:
            msg = (
                f"There were {len(self._unsafe_samples)} "
                "sample(s) that raised errors and were skipped."
            )
            if log is not None:
                with open(log, "w") as f:
                    [f.write(p + "\n") for p in self._unsafe_samples]
                msg += f"The unsafe file paths are logged in {log}"
            warnings.warn(msg)
        return self._unsafe_samples

    @property
    def is_index_built(self):
        """Returns True if all indices of the original dataset have been
        classified into safe_samples_indices or _unsafe_samples_indices.
        """
        return len(self.dataset) == len(self._safe_indices) + len(self._unsafe_indices)

    @property
    def num_samples_examined(self):
        return len(self._safe_indices) + len(self._unsafe_indices)

    def __len__(self):
        """Returns the length of the original dataset.
        NOTE: This is different from the number of actually valid samples.
        """
        return len(self.dataset)

    def __iter__(self):
        return (
            self._safe_get_item(i)[0]
            for i in range(len(self))
            if self._safe_get_item(i)[0] is not None
        )

    def __getitem__(self, idx):
        """If loading an index fails, behavior depends on self.unsafe_behavior

        self.unsafe_behavior = {
            "substitute": pick another sample,
            "raise": raise the error
            "none": return None
        """
        if self.unsafe_behavior == "substitute":
            attempts = 0
            while attempts < len(self.dataset):
                sample, caught_exception = self._safe_get_item(idx)
                if sample is not None:
                    return sample
                idx += 1
                attempts += 1
                idx = idx % len(self.dataset)  # loop around end to beginning
            raise IndexError(
                f"Tried all samples, none were safe. Last sample attempted raised: {caught_exception}"
            )
        elif self.unsafe_behavior == "raise" or self.unsafe_behavior == "none":
            try:
                sample = self.dataset[idx]
                if idx not in self._safe_indices:
                    self._safe_indices.append(idx)
                return sample
            except Exception:
                if idx not in self._unsafe_indices:
                    self._unsafe_indices.append(idx)
                if self.unsafe_behavior == "none":
                    return None
                else:  # raise the Exception
                    raise
        else:
            raise ValueError(
                f"unsafe_behavior must be 'substitute','raise', or 'none'. Got {self.unsafe_behavior}"
            )
