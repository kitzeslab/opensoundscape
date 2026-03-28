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

import copy
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

    Attributes: _valid_indices and _invalid_indices can be accessed later to check
    which samples raised Exceptions.

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
                - "substitute": pick another sample to load. Returned sample will have an attribute is_alternative = True
                - "placeholder": return a placeholder value (zeros) for the sample of the shape of the first
                    successfully loaded sample. Returned sample will have an attribute is_alternative = True
                    Note that if the first sample fails to load, an error will be raised since we don't have a placeholder
                    (unless user sets self.placeholder before accessing the first sample)
                - "raise": raise the error
                - "none": return None

        Returns:
            `SafeDataset` wrapper around `dataset`
        """

        self.dataset = dataset
        self.invalid_sample_behavior = invalid_sample_behavior
        self.placeholder = None  # set to the first successfully loaded sample

        # These lists will contain indices over the original dataset. The indices of
        # the safe samples will go into _valid_indices and similarly for invalid
        # samples in _invalid_indices.
        self._valid_indices = []
        self._invalid_indices = []

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
            if isinstance(exc, IndexError) and invalid_idx:
                # the index was out of bounds, so we actually want to raise this
                # IndexError instead of returning it
                raise
            if idx not in self._invalid_indices:
                self._invalid_indices.append(idx)

            # _return_ the exception (don't raise it)
            return exc

    def _reset_index(self):
        """Resets the valid and invalid samples indices, & invalid sample list."""
        self._valid_indices = []
        self._invalid_indices = []

    def report(self, log=None):
        """Warn about invalid samples and optionally write them to a log file

        Args:
            log: optional file path to save the list of invalid sample paths as CSV
                [default: None does not write a file]

        Returns:
            DataFrame (with no columns) whose index contains the paths of samples
            that raised errors during preprocessing
        """
        invalid_samples = self.dataset.label_df.iloc[self._invalid_indices][[]]
        if len(invalid_samples) > 0:
            msg = (
                f"There were {len(invalid_samples)} "
                "sample(s) that raised errors and were skipped or replaced."
            )

            if log is not None:
                invalid_samples.to_csv(log)
                msg += f"The invalid file paths are logged in {log}"
            warnings.warn(msg)
        return invalid_samples

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
            "substitute": pick another sample
            "placeholder": return a placeholder value (zeros) for the sample
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
                    # we successfully loaded a sample
                    if attempts > 0:
                        # mark the sample as a replacement
                        sample_or_exc.is_alternative = True
                    else:
                        # mark the sample as the original
                        sample_or_exc.is_alternative = False
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
        elif self.invalid_sample_behavior == "placeholder":
            # try to load the sample at idx, if it fails, return a placeholder sample
            # raises an error if the first sample fails to load and no placeholder is set
            sample_or_exc = self._safe_get_item(idx)
            if isinstance(sample_or_exc, Exception):
                # sample failed to load.
                # if we have a placeholder, return it
                # otherwise we failed to load the first sample, so raise the error

                if self.placeholder is None:
                    raise ValueError(
                        "The first sample requested from the dataset failed to load"
                    ) from sample_or_exc
                if idx not in self._invalid_indices:
                    self._invalid_indices.append(idx)
                return self.placeholder
            else:  # successfully loaded sample
                sample_or_exc.is_alternative = False
                if idx not in self._valid_indices:
                    self._valid_indices.append(idx)
                if self.placeholder is None:
                    # store this sample as the placeholder for future use
                    placeholder = copy.deepcopy(sample_or_exc)
                    # mark as not original sample (will fail if cannot set attributes on sample)
                    placeholder.is_alternative = True
                    try:  # try setting the values to zero, but allow failure
                        # in case the sample doesn't have .data or .labels attributes
                        placeholder.data = 0 * placeholder.data
                        placeholder.labels = 0 * placeholder.labels
                    except:
                        pass
                    self.placeholder = placeholder

                return sample_or_exc
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
                f"invalid_sample_behavior must be 'substitute','placeholder', 'raise', or 'none'. "
                f"Got {self.invalid_sample_behavior}"
            )
