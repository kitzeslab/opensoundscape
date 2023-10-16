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
    `.dataset` attribute that is a DataSet (or AudioFileDataset / AudioSplittingDataset,
    which subclass DataSet).

    Args:
        dataset: a torch Dataset instance or child such as AudioFileDataset, AudioSplittingDataset
        eager_eval: If True, checks if every file is able to be loaded during
            initialization (logs _valid_indices and _invalid_indices)

    Attributes: _vlid_indices and _invalid_indices can be accessed later to check
    which samples raised Exceptions. _invalid_samples is a set of all index values
    for samples that raised Exceptions.

    Methods:
        __getitem__(index):
            If loading an index fails, keeps trying the next index until success
        _safe_get_item():
            Tries to load a sample, returns None if error occurs
        __iter__():
            generator that skips samples that raise errors when loading
    """

    def __init__(self, dataset, invalid_sample_behavior):
        """Creates a `SafeDataset` wrapper on a DataSet to handle bad samples

        Args:
            dataset: a Pytorch DataSet object
            invalid_sample_behavior: what to do when loading a sample results in an error
                - "substitute": pick another sample to load
                - "raise": raise the error
                - "none": return None

        Returns:
            `SafeDataset` wrapper around `dataset`
        """

        self.dataset = dataset
        self.invalid_sample_behavior = invalid_sample_behavior
        # These will contain indices over the original dataset. The indices of
        # the safe samples will go into _valid_indices and similarly for invalid
        # samples in _invalid_indices. _invalid_samples holds the actual names
        self._valid_indices = []
        self._invalid_indices = []
        self._invalid_samples = set()

    def _safe_get_item(self, idx):
        """attempts to load sample at idx, returns the exception if it fails

        Returns an Exception (instead of raising) it when any exception is raised
        while trying to load a sample from a dataset

        and also builds an index of valid and invalid samples as
        and when they get accessed.

        Args:
            idx: index of sample to load (same as torch Dataset's __getitem__)

        Returns:
            the sample at idx, or the Exception that was raised
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
            # the index was out of bounds, so we actually want to raise this
            # IndexError
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

            # _return_ the exception (don't raise it)
            return exc

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
                        f.write(f"{p} \n")
                msg += f"The invalid file paths are logged in {log}"
            warnings.warn(msg)
        return self._invalid_samples

    def __len__(self):
        """Returns the length of the original dataset.
        NOTE: This is different from the number of actually valid samples.
        """
        return len(self.dataset)

    def __iter__(self):
        """generator: skips samples that raised errors when loading"""
        return (
            self._safe_get_item(i)
            for i in range(len(self))
            if not isinstance(self._safe_get_item(i), Exception)
        )

    def __getitem__(self, idx):
        """If loading an index fails, behavior depends on self.invalid_sample_behavior

        self.invalid_sample_behavior:
            "substitute": pick another sample,
            "raise": raise the error
            "none": return None
        """
        if self.invalid_sample_behavior == "substitute":
            # try to load the sample at idx, if it fails, try the next sample
            # until we find one that works or have tried all samples
            attempts = 0
            sample_or_exc = Exception  # placeholder
            while attempts < len(self.dataset):
                sample_or_exc = self._safe_get_item(idx)
                if not isinstance(sample_or_exc, Exception):
                    return sample_or_exc
                idx += 1
                attempts += 1
                idx = idx % len(self.dataset)  # loop around end to beginning

            # raises _from_ the most recent exception during preprocessing
            raise IndexError(
                "None of the samples in the SafeDataset loaded. "
                "All samples caused exceptions during preprocessing. The most"
                "recent exception is in the error trace. "
            ) from sample_or_exc
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
