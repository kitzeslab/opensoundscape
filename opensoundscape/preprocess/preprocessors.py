import pandas as pd
import numpy as np
import torch
from opensoundscape.preprocess import actions
from opensoundscape.preprocess.actions import ParameterRequiredError
from pathlib import Path


class BasePreprocessor(torch.utils.data.Dataset):
    """Base class for Preprocessing pipelines (use in place of torch Dataset)

    Custom Preprocessor classes should subclass this class or its children

    df must have audio paths as index and class names as columns
    - if no labels, df will have no columns
    """

    def __init__(self, df, return_labels=True):
        # TODO: add .sample method?

        assert Path(df.index[0]).exists(), (
            "Index of dataframe passed to "
            f"preprocessor must be a file path. Got {df.index[0]}."
        )
        assert df.values[0, 0] in (0, 1)

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

        df_row = self.df.iloc[item_idx]
        x = Path(df_row.name)  # the index contains a path to a file

        for pipeline_element in self.pipeline:
            if pipeline_element.bypass:
                continue
            try:
                x = pipeline_element.go(x)
            except ParameterRequiredError:  # need to pass labels
                x, df_row = pipeline_element.go(x, df_row)

        # Return sample & label pairs (training/validation)
        if self.return_labels:
            labels = torch.from_numpy(df_row.values)
            return {"X": x, "y": labels}

        # Return sample only (prediction)
        return {"X": x}

    def class_counts_cal(self):
        """count number of each label"""
        labels = self.df.columns
        counts = np.sum(self.df.values, 0)
        return labels, counts


class AudioLoadingPreprocessor(BasePreprocessor):
    """creates Audio objects from file paths

    df must have audio paths as index and class names as columns
    - if no labels, df will have no columns
    """

    def __init__(self, df, return_labels=True):

        super(AudioLoadingPreprocessor, self).__init__(df, return_labels=return_labels)

        # add an AudioLoader to our (currently empty) action toolkit
        self.actions.load_audio = actions.AudioLoader()

        # add the action to our (currently empty) pipeline
        self.pipeline.append(self.actions.load_audio)

        # add a second action for trimming audio (default is no trimming)
        self.actions.trim_audio = actions.AudioTrimmer()
        self.pipeline.append(self.actions.trim_audio)


class AudioToSpectrogramPreprocessor(BasePreprocessor):
    """loads audio paths, creates spectrogram, returns tensor

    by default, resamples audio to sr=22050
    can change with .actions.load_audio.set(sample_rate=sr)

    df must have audio paths as index and class names as columns
    - if no labels, df will have no columns
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

        self.actions.trim_audio = actions.AudioTrimmer()
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
    """Child of AudioToSpectrogramPreprocessor with full augmentation pipeline"""

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
        # TODO: should be able to create object without overlay_df
        self.actions.overlay = actions.ImgOverlay(
            overlay_df=overlay_df,
            audio_length=self.audio_length,
            overlay_prob=1,
            max_overlay_num=1,
            overlay_class=None,
            # TODO: check - overlay pipeline might not update with changes?
            loader_pipeline=self.pipeline[0:5],  # all actions before overlay
            update_labels=False,
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

        if self.debug is not None:
            self.actions.save_img = actions.SaveTensorToDisk(self.debug)
            self.pipeline.append(self.actions.save_img)

    def augmentation_on():
        """use pipeline containing all actions including augmentations"""
        self.pipeline = self.augmentation_pipeline

    def augmentation_off():
        """use pipeline that skips all augmentations"""
        self.pipeline = self.no_augmentation_pipeline


class ResnetMultilabelPreprocessor(BasePreprocessor):
    """loads audio paths, performs various augmentations, returns tensor"""

    def __init__(
        self,
        df,
        audio_length=None,
        return_labels=True,
        augmentation=True,
        debug=None,
        overlay_df=None,
    ):

        super(ResnetMultilabelPreprocessor, self).__init__(
            df, return_labels=return_labels
        )

        self.audio_length = audio_length
        self.augmentation = augmentation
        self.return_labels = return_labels
        self.debug = debug

        # add each action to our tool kit, then to pipeline
        self.actions.load_audio = actions.AudioLoader()
        self.pipeline.append(self.actions.load_audio)

        self.actions.trim_audio = actions.AudioTrimmer()
        self.pipeline.append(self.actions.trim_audio)

        self.actions.to_spec = actions.AudioToSpectrogram()
        self.pipeline.append(self.actions.to_spec)

        self.actions.to_img = actions.SpecToImg()
        self.pipeline.append(self.actions.to_img)

        # should make one without overlay, then subclass and add overlay
        if self.augmentation:
            self.actions.overlay = actions.ImgOverlay(
                overlay_df=overlay_df,
                audio_length=self.audio_length,
                overlay_prob=0.5,
                max_overlay_num=2,
                overlay_weight=[0.2, 0.5],
                # this pipeline might not update with changes to preprocessor?
                loader_pipeline=self.pipeline[0:4],
                update_labels=False,
            )
            self.pipeline.append(self.actions.overlay)

            # color jitter and affine can be applied to img or tensor
            # here, we choose to apply them to the PIL.Image
            self.actions.color_jitter = actions.TorchColorJitter()
            self.pipeline.append(self.actions.color_jitter)

            self.actions.random_affine = actions.TorchRandomAffine()
            self.pipeline.append(self.actions.random_affine)

        self.actions.to_tensor = actions.ImgToTensor()
        self.pipeline.append(self.actions.to_tensor)

        if self.augmentation:
            self.actions.tensor_aug = actions.TensorAugment()
            self.pipeline.append(self.actions.tensor_aug)

            self.actions.add_noise = actions.TensorAddNoise(std=1.0)
            self.pipeline.append(self.actions.add_noise)

        self.actions.normalize = actions.TensorNormalize()
        self.pipeline.append(self.actions.normalize)

        if self.debug is not None:
            self.actions.save_img = actions.SaveTensorToDisk(self.debug)
            self.pipeline.append(self.actions.save_img)
