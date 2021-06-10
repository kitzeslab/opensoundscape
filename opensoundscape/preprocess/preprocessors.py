import pandas as pd
import numpy as np
import torch
from opensoundscape.preprocess import actions
from pathlib import Path
import copy
from opensoundscape.preprocess.utils import PreprocessingError


class BasePreprocessor(torch.utils.data.Dataset):
    """Base class for Preprocessing pipelines (use in place of torch Dataset)

    Custom Preprocessor classes should subclass this class or its children

    Args:
        df:
            dataframe of samples. df must have audio paths in the index.
            If df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, df will have no columns
        return_labels:
            if True, the __getitem__ method will return {X:sample,y:labels}
            If False, the __getitem__ method will return {X:sample}
            If df has no labels (no columns), use return_labels=False
            [default: True]

    Raises:
        PreprocessingError if exception is raised during __getitem__
    """

    def __init__(self, df, return_labels=True):

        assert Path(df.index[0]).exists(), (
            "Index of dataframe passed to "
            f"preprocessor must be a file path. Got {df.index[0]}."
        )
        if return_labels:
            assert df.values[0, 0] in (
                0,
                1,
            ), "if return_labels=True, df must have labels that take values of 0 and 1"

        self.df = df
        self.return_labels = return_labels
        self.labels = df.columns

        # actions: a collection of instances of BaseAction child classes
        self.actions = actions.ActionContainer()

        # pipeline: an ordered list of operations to conduct on each file,
        # each pulled from self.actions
        self.pipeline = []

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item_idx):

        try:
            df_row = self.df.iloc[item_idx]
            x = Path(df_row.name)  # the index contains a path to a file

            # perform each action in the pipeline if action.bypass==False
            for action in self.pipeline:
                if action.bypass:
                    continue
                if action.requires_labels:
                    x, df_row = action.go(x, copy.deepcopy(df_row))
                else:
                    x = action.go(x)

            # Return sample & label pairs (training/validation)
            if self.return_labels:
                labels = torch.from_numpy(df_row.values)
                return {"X": x, "y": labels}

            # Return sample only (prediction)
            return {"X": x}
        except:
            raise PreprocessingError(
                f"failed to preprocess sample: {self.df.index[item_idx]}"
            )

    def class_counts_cal(self):
        """count number of each label"""
        labels = self.df.columns
        counts = np.sum(self.df.values, 0)
        return labels, counts

    def sample(self, **kwargs):
        """out-of-place random sample

        creates copy of object with n rows randomly sampled from dataframe

        Args: see pandas.DataFrame.sample()

        Returns:
            a new dataset object
        """
        new_ds = copy.deepcopy(self)
        new_ds.df = new_ds.df.sample(**kwargs)
        return new_ds

    def head(self, n=5):
        """out-of-place copy of first n samples

        performs df.head(n) on self.df

        Args:
            n: number of first samples to return, see pandas.DataFrame.head()
            [default: 5]

        Returns:
            a new dataset object
        """
        new_ds = copy.deepcopy(self)
        new_ds.df = new_ds.df.head(n)
        return new_ds


class AudioLoadingPreprocessor(BasePreprocessor):
    """creates Audio objects from file paths

    Args:
        df:
            dataframe of samples. df must have audio paths in the index.
            If df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, df will have no columns
        return_labels:
            if True, __getitem__ returns {"X":batch_tensors,"y":labels}
            if False, __getitem__ returns {"X":batch_tensors}
            [default: True]
        audio_length:
            length in seconds of audio to return
            - None: do not trim the original audio
            - seconds (float): trim longer audio to this length. Shorter
            audio input will raise a ValueError.
    """

    def __init__(self, df, return_labels=True, audio_length=None):

        super(AudioLoadingPreprocessor, self).__init__(df, return_labels=return_labels)

        # add an AudioLoader to our (currently empty) action toolkit
        self.actions.load_audio = actions.AudioLoader()

        # add the action to our (currently empty) pipeline
        self.pipeline.append(self.actions.load_audio)

        # add a second action for trimming audio (default is no trimming)
        self.actions.trim_audio = actions.AudioTrimmer(
            extend_short_clips=False, random_trim=False, audio_length=audio_length
        )
        self.pipeline.append(self.actions.trim_audio)


class AudioToSpectrogramPreprocessor(BasePreprocessor):
    """
    loads audio paths, creates spectrogram, returns tensor

    by default, resamples audio to sr=22050
    can change with .actions.load_audio.set(sample_rate=sr)

    Args:
        df:
            dataframe of samples. df must have audio paths in the index.
            If df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, df will have no columns
        audio_length:
            length in seconds of audio clips [default: None]
            If provided, longer clips trimmed to this length. By default,
            shorter clips will not be extended (modify actions.AudioTrimmer
            to change behavior).
        out_shape:
            output shape of tensor in pixels [default: [224,224]]
        return_labels:
            if True, the __getitem__ method will return {X:sample,y:labels}
            If False, the __getitem__ method will return {X:sample}
            If df has no labels (no columns), use return_labels=False
            [default: True]
    """

    def __init__(self, df, audio_length=None, out_shape=[224, 224], return_labels=True):

        super(AudioToSpectrogramPreprocessor, self).__init__(
            df, return_labels=return_labels
        )

        self.audio_length = audio_length
        self.return_labels = return_labels

        # add each action to our tool kit, then to pipeline
        self.actions.load_audio = actions.AudioLoader(sample_rate=22050)
        self.pipeline.append(self.actions.load_audio)

        self.actions.trim_audio = actions.AudioTrimmer(
            extend_short_clips=False, random_trim=False, audio_length=audio_length
        )
        self.pipeline.append(self.actions.trim_audio)

        self.actions.to_spec = actions.AudioToSpectrogram()
        self.pipeline.append(self.actions.to_spec)

        self.actions.bandpass = actions.SpectrogramBandpass(min_f=0, max_f=20000)
        self.pipeline.append(self.actions.bandpass)
        self.actions.bandpass.off()  # bandpass is off by default

        self.actions.to_img = actions.SpecToImg(shape=out_shape)
        self.pipeline.append(self.actions.to_img)

        self.actions.to_tensor = actions.ImgToTensor()
        self.pipeline.append(self.actions.to_tensor)

        self.actions.normalize = actions.TensorNormalize()
        self.pipeline.append(self.actions.normalize)


class CnnPreprocessor(AudioToSpectrogramPreprocessor):
    """Child of AudioToSpectrogramPreprocessor with full augmentation pipeline

    loads audio, creates spectrogram, performs augmentations, returns tensor

    by default, resamples audio to sr=22050
    can change with .actions.load_audio.set(sample_rate=sr)

    Args:
        df:
            dataframe of samples. df must have audio paths in the index.
            If df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, df will have no columns
        audio_length:
            length in seconds of audio clips [default: None]
            If provided, longer clips trimmed to this length. By default,
            shorter clips will not be extended (modify actions.AudioTrimmer
            to change behavior).
        out_shape:
            output shape of tensor in pixels [default: [224,224]]
        return_labels:
            if True, the __getitem__ method will return {X:sample,y:labels}
            If False, the __getitem__ method will return {X:sample}
            If df has no labels (no columns), use return_labels=False
            [default: True]
        debug:
            If a path is provided, generated samples (after all augmentation)
            will be saved to the path as an image. This is useful for checking
            that the sample provided to the model matches expectations.
            [default: None]
    """

    def __init__(
        self,
        df,
        audio_length=None,
        return_labels=True,
        debug=None,
        overlay_df=None,
        out_shape=[224, 224],
    ):

        super(CnnPreprocessor, self).__init__(
            df,
            audio_length=audio_length,
            out_shape=out_shape,
            return_labels=return_labels,
        )

        self.debug = debug

        # extra Actions for augmentation steps
        self.actions.overlay = (
            actions.ImgOverlay(
                overlay_df=overlay_df,
                audio_length=self.audio_length,
                overlay_prob=1,
                max_overlay_num=1,
                overlay_class=None,
                loader_pipeline=self.pipeline[0:5],  # all actions up to overlay
                update_labels=False,
            )
            if overlay_df is not None
            else actions.BaseAction()
        )

        self.actions.color_jitter = actions.TorchColorJitter()
        self.actions.random_affine = actions.TorchRandomAffine()
        self.actions.time_mask = actions.TimeMask()
        self.actions.frequency_mask = actions.FrequencyMask()
        # self.actions.time_warp = actions.TimeWarp()
        self.actions.add_noise = actions.TensorAddNoise(std=0.005)

        self.augmentation_pipeline = [
            self.actions.load_audio,
            self.actions.trim_audio,
            self.actions.to_spec,
            self.actions.bandpass,
            self.actions.to_img,
            self.actions.overlay,
            self.actions.color_jitter,
            self.actions.to_tensor,
            # self.actions.time_warp,
            self.actions.time_mask,
            self.actions.frequency_mask,
            self.actions.add_noise,
            self.actions.normalize,
            self.actions.random_affine,
        ]

        self.no_augmentation_pipeline = [
            self.actions.load_audio,
            self.actions.trim_audio,
            self.actions.to_spec,
            self.actions.bandpass,
            self.actions.to_img,
            self.actions.to_tensor,
            self.actions.normalize,
        ]

        self.pipeline = self.augmentation_pipeline

        if self.debug is not None:
            self.actions.save_img = actions.SaveTensorToDisk(self.debug)
            self.pipeline.append(self.actions.save_img)

    def augmentation_on(self):
        """use pipeline containing all actions including augmentations"""
        self.pipeline = self.augmentation_pipeline

    def augmentation_off(self):
        """use pipeline that skips all augmentations"""
        self.pipeline = self.no_augmentation_pipeline
