import random
import warnings
import numpy as np
import pandas as pd
import torchvision

from opensoundscape.preprocess.actions import (
    register_action_cls,
    Action,
)
from opensoundscape.sample import AudioSample
from opensoundscape.preprocess.utils import PreprocessingError
from opensoundscape.preprocess.actions import register_action_fn


@register_action_cls
class Overlay(Action):
    """Action Class for augmentation that overlays samples on eachother

    Overlay is a flavor of "mixup" augmentation, where two samples are
    overlayed on top of eachother. The samples are blended with a weighted
    average, where the weight may be chosen randomly from a range of values.

    In this implementation, the overlayed samples are chosen from a dataframe
    of audio files and labels. The dataframe must have the audio file paths as
    the index, and the labels as columns. The labels are used to choose
    overlayed samples based on an "overlay_class" argument.

    Args:
        overlay_df: dataframe of audio files (index) and labels to use for overlay
        update_labels (bool): if True, labels of sample are updated to include
            labels of overlayed sample
        criterion_fn: function that takes AudioSample and returns True or False
            - if True, perform overlay
            - if False, do not perform overlay
            Default is `always_true`, perform overlay on all samples

        See overlay() for **kwargs and default values

    """

    def __init__(self, is_augmentation=True, **kwargs):
        if "fn" in kwargs:
            kwargs.pop("fn")
        super(Overlay, self).__init__(
            overlay,
            is_augmentation=is_augmentation,
            **kwargs,
        )

        # self.returns_labels = True

        overlay_df = kwargs["overlay_df"].copy()  # copy to avoid modifying original
        overlay_df = overlay_df[~overlay_df.index.duplicated()]  # remove duplicates

        # warn the user if using "different" as overlay_class
        # and "different" is one of the model classes
        if (
            "different" in overlay_df.columns
            and "overlay_class" in kwargs
            and kwargs["overlay_class"] == "different"
        ):
            warnings.warn(
                "class name `different` was in columns, but using "
                "kwarg overlay_class='different' has specific behavior and will "
                "not specifically choose files from the `different` class. "
                "Consider renaming the `different` class. "
            )

        # move overlay_df from .params to its own attribute
        self.overlay_df = overlay_df
        self.params = self.params.drop("overlay_df")  # removes it from params Series

    def __call__(self, sample, **kwargs):
        self.action_fn(
            sample,
            overlay_df=self.overlay_df,
            **dict(self.params, **kwargs),
        )

    def to_dict(self):
        # don't save self.overlay_df since it might be huge and is not json friendly
        # also don't save criterion_fn, will default to always_true on reload
        # user will have to specify these after using from_dict
        d = super().to_dict(ignore_attributes=("overlay_df"))
        d["params"].pop("criterion_fn")
        return d

    @classmethod
    def from_dict(cls, dict):
        # two attributes of the Overlay class are not json-friendly and not saved with to_dict;
        # we instead initialize with an empty overlay_df and set criterion_fn to always_true
        # we also initialize the Overlay action with bypass=True so that it is inactive by default
        dict["params"]["overlay_df"] = pd.DataFrame()
        dict["params"]["criterion_fn"] = always_true
        instance = super().from_dict(dict)
        instance.bypass = True
        warnings.warn(
            "Overlay class's .overlay_df will be None after loading from dict and `.criterion_fn` will be always_true(). "
            "Reset these attributes and set .bypass to False to use Overlay after loading with from_dict()."
        )
        return instance


def always_true(x):
    return True


@register_action_fn
def overlay(
    sample,
    overlay_df,
    update_labels,
    overlay_class=None,
    overlay_prob=1,
    max_overlay_num=1,
    overlay_weight=0.5,
    criterion_fn=always_true,
):
    """iteratively overlay 2d samples on top of eachother

    Overlays (blends) image-like samples from overlay_df on top of
    the sample with probability `overlay_prob` until stopping condition.
    If necessary, trims overlay audio to the length of the input audio.

    Optionally provide `criterion_fn` which takes sample and returns True/False
    to determine whether to perform overlay on this sample.

    Overlays can be used in a few general ways:
        1. a separate df where any file can be overlayed (overlay_class=None)
        2. same df as training, where the overlay class is "different" ie,
            does not contain overlapping labels with the original sample
        3. same df as training, where samples from a specific class are used
            for overlays

    Args:
        sample: AudioSample with .labels: labels of the original sample
            and .preprocessor: the preprocessing pipeline
        overlay_df: a labels dataframe with audio files as the index and
            classes as columns

        update_labels: if True, add overlayed sample's labels to original sample
        overlay_class: how to choose files from overlay_df to overlay
            Options [default: None]:
            None - Randomly select any file from overlay_df
            "different" - Select a random file from overlay_df containing none
                of the classes this file contains
            specific class name - always choose files from this class
        overlay_prob: the probability of applying each subsequent overlay
        max_overlay_num: the maximum number of samples to overlay on original
            - for example, if overlay_prob = 0.5 and max_overlay_num=2,
                1/2 of samples will recieve 1 overlay and 1/4 will recieve an
                additional second overlay
        overlay_weight: a float > 0 and < 1, or a list of 2 floats [min, max]
            between which the weight will be randomly chosen. e.g. [0.1,0.7]
            An overlay_weight <0.5 means more emphasis on original sample.
        criterion_fn: function that takes AudioSample and returns True or False
            - if True, perform overlay
            - if False, do not perform overlay
            Default is `always_true`, perform overlay on all samples

    Returns:
        overlayed sample, (possibly updated) labels


    Example:
        check if sample is from a xeno canto file (has "XC" in name),
        and only perform overlay on xeno canto files
        ```
        def is_xc(audio_sample):
            return "XC" in Path(audio_sample.source).stem
        s=overlay(s, overlay_df, False, criterion_fn=is_xc)
        ```
    """

    # Use the criterion_fn to determine if we should perform overlay on this sample
    if not criterion_fn(sample):
        return sample  # no overlay, just return the original sample

    ##  INPUT VALIDATION ##
    assert (
        overlay_class in ["different", None] or overlay_class in overlay_df.columns
    ), (
        "overlay_class must be 'different' or None or in overlay_df.columns. "
        f"got {overlay_class}"
    )
    assert (overlay_prob <= 1) and (overlay_prob >= 0), (
        "overlay_prob" f"should be in range (0,1), was {overlay_prob}"
    )

    weight_error = f"overlay_weight should be between 0 and 1, was {overlay_weight}"

    if hasattr(overlay_weight, "__iter__"):
        assert (
            len(overlay_weight) == 2
        ), "must provide a float or a range of min,max values for overlay_weight"
        assert (
            overlay_weight[1] > overlay_weight[0]
        ), "second value must be greater than first for overlay_weight"
        for w in overlay_weight:
            assert w < 1 and w > 0, weight_error
    else:
        assert overlay_weight < 1 and overlay_weight > 0, weight_error

    if overlay_class is not None:
        assert (
            len(overlay_df.columns) > 0
        ), "overlay_df must have labels if overlay_class is specified"
        if overlay_class != "different":  # user specified a single class
            assert (
                overlay_df[overlay_class].sum() > 0
            ), "overlay_df did not contain positive labels for overlay_class"

    if len(overlay_df.columns) > 0 and sample.labels is not None:
        assert list(overlay_df.columns) == list(
            sample.labels.index
        ), "overlay_df mast have same columns as sample's _labels or no columns"

    ## OVERLAY ##
    # iteratively perform overlays until stopping condition
    # each time, there is an overlay_prob probability of another overlay
    # up to a max number of max_overlay_num overlays
    overlays_performed = 0

    while overlay_prob > np.random.uniform() and overlays_performed < max_overlay_num:
        try:
            # lets pick a sample based on rules
            if overlay_class is None:
                # choose any file from the overlay_df
                overlay_path = random.choice(overlay_df.index)

            elif overlay_class == "different":
                # Select a random file containing none of the classes this file contains
                # because the overlay_df might be huge and sparse, we randomly
                # choose row until one fits criterea rather than filtering overlay_df
                # TODO: revisit this choice
                good_choice = False
                attempt_counter = 0
                max_attempts = 100  # if we try this many times, raise error
                while (not good_choice) and (attempt_counter < max_attempts):
                    attempt_counter += 1

                    # choose a random sample from the overlay df
                    candidate_idx = random.randint(0, len(overlay_df) - 1)

                    # check if this candidate sample has zero overlapping labels
                    label_intersection = np.logical_and(
                        overlay_df.values[candidate_idx, :], sample.labels.values
                    )
                    good_choice = sum(label_intersection) == 0

                if not good_choice:  # tried max_attempts samples, none worked
                    raise ValueError(
                        f"No samples found with non-overlapping labels after {max_attempts} random draws"
                    )

                overlay_path = overlay_df.index[candidate_idx]

            else:
                # Select a random file from a class of choice (may be slow -
                # however, in the case of a fixed overlay class, we could
                # pass an overlay_df containing only that class)
                choose_from = overlay_df[overlay_df[overlay_class] == 1]
                overlay_path = np.random.choice(choose_from.index.values)

            # now we have picked a file to overlay (overlay_path)
            # we also know its labels, if we need them
            # TODO: this will be slow with large index but fast with numeric index, reset_index() somewhere
            overlay_sample = AudioSample.from_series(overlay_df.loc[overlay_path])

            # now we need to run the pipeline to do everything up until the Overlay step
            # create a preprocessor for loading the overlay samples
            # note that if there are multiple Overlay objects in a pipeline,
            # it will cut off the preprocessing of the overlayed sample before
            # the first Overlay object. This may or may not be the desired behavior,
            # but it will at least "work".
            overlay_sample = sample.preprocessor.forward(
                overlay_sample, break_on_type=Overlay
            )

            # the overlay_sample may have a different shape than the original sample
            # force them into the same shape so we can overlay
            if overlay_sample.data.shape != sample.data.shape:
                overlay_sample.data = torchvision.transforms.Resize(
                    sample.data.shape[1:]
                )(overlay_sample.data)

            # now we blend the two tensors together with a weighted average
            # Select weight of overlay; <0.5 means more emphasis on original sample
            # Supports uniform-random selection from a range of weights eg [0.1,0.7]
            weight = overlay_weight
            if hasattr(weight, "__iter__"):
                assert len(weight) == 2, (
                    f"overlay_weight must specify a single value or range of 2 values, "
                    f"got {overlay_weight}"
                )
                weight = random.uniform(weight[0], weight[1])

            # use a weighted sum to overlay (blend) the samples (arrays or tensors)
            sample.data = sample.data * (1 - weight) + overlay_sample.data * weight

            # update the labels with new classes
            if update_labels and len(overlay_sample.labels) > 0:
                # update labels as union of both files' labels
                sample.labels.values[:] = np.logical_or(
                    sample.labels.values, overlay_sample.labels.values
                ).astype(int)

            # overlay was successful, update count:
            overlays_performed += 1

        except PreprocessingError as ex:
            # don't try to load this sample again: remove from overlay df
            overlay_df = overlay_df.drop(overlay_path)
            warnings.warn(f"Invalid overlay sample: {overlay_path}")
            if len(overlay_df) < 1:
                raise ValueError("tried all overlay_df samples, none were safe") from ex

    return sample
