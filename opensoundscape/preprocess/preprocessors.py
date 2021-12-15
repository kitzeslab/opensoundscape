import pandas as pd
import numpy as np
import torch
from deprecated import deprecated
from pathlib import Path
import copy

from opensoundscape.preprocess.utils import PreprocessingError
from opensoundscape.preprocess import actions


class BasePreprocessor(torch.utils.data.Dataset):
    """Base class for Preprocessing pipelines (use in place of torch Dataset)

    Custom Preprocessor classes should subclass this class or its children

    Args:
        df:
            dataframe of audio clips. df must have audio paths in the index.
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
        self.classes = df.columns
        self.specifies_clip_times = False

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

    def pipeline_summary(self):
        """Generate a DataFrame describing the current pipeline

        The DataFrame has columns for name (corresponds to the attribute
        name, eg 'to_img' for self.actions.to_img), on (not bypassed) / off
        (bypassed), and action_reference (a reference to the object)
        """
        df = pd.DataFrame(columns=["name", "on/off", "action_reference"])
        action_dict = {val: key for key, val in vars(self.actions).items()}

        for action in self.pipeline:
            df = df.append(
                {
                    "name": action_dict[action],
                    "on/off": "off" if action.bypass else "ON",
                    "action_reference": action,
                },
                ignore_index=True,
            )

        return df


class AudioLoadingPreprocessor(BasePreprocessor):
    """creates Audio objects from file paths

    Args:
        df:
            dataframe of audio clips. df must have audio paths in the index.
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


# basically we need a class that's equivalent to the regular one, but
# allows the input df to have additional indices 'start_time','end_time'
# or, instead, could have additional dataframe input listing clips - then
# might be able to use all the same classes. Like,
# BasePreprocessor(df....)
# if df.index is (path, start_t, end_t)...
# otherwise, assert df.index is path...
# but then every action would need to handle the possibility of
# the nice thing is that this would integrate well with labels
# for instance label_df created from long raven file, with triple-index.
# the only actions (currently) that require labels/use file path are
# overlay and save tensor.


@deprecated(
    version="0.6.1",
    reason="Use ClipLoadingSpectrogramPreprocessor"
    "for similar functionality with lower memory requirements.",
)
class LongAudioPreprocessor(BasePreprocessor):
    """
    loads audio paths, splits into segments, and runs pipeline on each segment

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
        clip_overlap:
            overlap in seconds between adjacent clips
        final_clip=None:
            see Audio.split() for final clip behavior and options
        out_shape:
            output shape of tensor in pixels [default: [224,224]]

        Note: returning labels is not implemented

    Returns: DataSet object
    """

    def __init__(
        self, df, audio_length, clip_overlap=0, final_clip=None, out_shape=[224, 224]
    ):

        super(LongAudioPreprocessor, self).__init__(df, return_labels=False)

        self.audio_length = audio_length
        self.clip_overlap = clip_overlap
        self.final_clip = final_clip

        # create separate pipeline for any actions that happen before audio splitting
        # running the actions of pre_split_pipeline should result in a single audio object
        self.pre_split_pipeline = []
        self.actions.load_audio = actions.AudioLoader(sample_rate=None)
        self.pre_split_pipeline.append(self.actions.load_audio)

        # add each action to our tool kit, then to pipeline
        self.actions.trim_audio = actions.AudioTrimmer(
            extend=True, random_trim=False, audio_length=audio_length
        )
        self.pipeline.append(self.actions.trim_audio)

        self.actions.to_spec = actions.AudioToSpectrogram()
        self.pipeline.append(self.actions.to_spec)

        # bandpass since we don't resample audio to guarantee equivalence
        self.actions.bandpass = actions.SpectrogramBandpass(min_f=0, max_f=11025)
        self.pipeline.append(self.actions.bandpass)

        self.actions.to_img = actions.SpecToImg(shape=out_shape)
        self.pipeline.append(self.actions.to_img)

        self.actions.to_tensor = actions.ImgToTensor()
        self.pipeline.append(self.actions.to_tensor)

        self.actions.normalize = actions.TensorNormalize()
        self.pipeline.append(self.actions.normalize)

    def __getitem__(self, item_idx):

        try:
            df_row = self.df.iloc[item_idx]
            x = Path(df_row.name)  # the index contains a path to a file

            # First, run the pre_split_pipeline to get an audio object
            for action in self.pre_split_pipeline:
                if action.bypass:
                    continue
                if action.requires_labels:
                    x, df_row = action.go(x, copy.deepcopy(df_row))
                else:
                    x = action.go(x)

            # Second, split the audio
            audio_clips, clip_df = x.split(
                clip_duration=self.audio_length,
                clip_overlap=self.clip_overlap,
                final_clip=self.final_clip,
            )
            if len(clip_df) < 1:
                raise ValueError(f"File produced no samples: {Path(df_row.name)}")

            clip_df = pd.DataFrame(
                index=[
                    [df_row.name] * len(clip_df),
                    clip_df["start_time"],
                    clip_df["end_time"],
                ]
            )
            clip_df.index.names = ["file", "start_time", "end_time"]

            # Third, for each audio segment, run the rest of the pipeline and store the output
            outputs = [None for _ in range(len(audio_clips))]
            for i, x in enumerate(audio_clips):
                for action in self.pipeline:
                    if action.bypass:
                        continue
                    if action.requires_labels:
                        x, df_row = action.go(x, copy.deepcopy(df_row))
                    else:
                        x = action.go(x)
                outputs[i] = x

            # concatenate the outputs into a single tensor
            if type(outputs[0]) == torch.Tensor:
                outputs = torch.Tensor(np.stack(outputs))

            # Return sample & label pairs (training/validation)
            if self.return_labels:
                labels = torch.from_numpy(df_row.values)
                return {"X": outputs, "y": labels, "df": clip_df}

            # Return sample only (prediction)
            return {"X": outputs, "df": clip_df}
        except:
            raise PreprocessingError(
                f"failed to preprocess sample: {self.df.index[item_idx]}"
            )


class AudioToSpectrogramPreprocessor(BasePreprocessor):
    """
    loads audio paths, creates spectrogram, returns tensor

    by default, does not resample audio, but bandpasses to 0-11025 Hz
    (to ensure all outputs have same scale in y-axis)
    can change with .actions.load_audio.set(sample_rate=sr)

    Args:
        df:
            dataframe of audio clips. df must have audio paths in the index.
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
        self.actions.load_audio = actions.AudioLoader(sample_rate=None)
        self.pipeline.append(self.actions.load_audio)

        self.actions.trim_audio = actions.AudioTrimmer(
            extend_short_clips=False, random_trim=False, audio_length=audio_length
        )
        self.pipeline.append(self.actions.trim_audio)

        self.actions.to_spec = actions.AudioToSpectrogram()
        self.pipeline.append(self.actions.to_spec)

        self.actions.bandpass = actions.SpectrogramBandpass(
            min_f=0, max_f=11025, out_of_bounds_ok=False
        )
        self.pipeline.append(self.actions.bandpass)

        self.actions.to_img = actions.SpecToImg(shape=out_shape)
        self.pipeline.append(self.actions.to_img)

        self.actions.to_tensor = actions.ImgToTensor()
        self.pipeline.append(self.actions.to_tensor)

        self.actions.normalize = actions.TensorNormalize()
        self.pipeline.append(self.actions.normalize)


class CnnPreprocessor(AudioToSpectrogramPreprocessor):
    """Child of AudioToSpectrogramPreprocessor with full augmentation pipeline

    loads audio, creates spectrogram, performs augmentations, returns tensor

    by default, does not resample audio, but bandpasses to 0-10 kHz
    (to ensure all outputs have same scale in y-axis)
    can change with .actions.load_audio.set(sample_rate=sr)

    Args:
        df:
            dataframe of audio clips. df must have audio paths in the index.
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

        # overlay
        if overlay_df is not None:
            # first, remove duplicate indices from overlay_df
            overlay_df = overlay_df[~overlay_df.index.duplicated()]

            # create overlay action
            self.actions.overlay = actions.ImgOverlay(
                overlay_df=overlay_df,
                audio_length=self.audio_length,
                overlay_prob=1,
                max_overlay_num=1,
                overlay_class=None,
                loader_pipeline=self.pipeline[0:5],  # all actions up to overlay
                update_labels=False,
            )
        else:  # create a blank action instead
            self.actions.overlay = actions.BaseAction()

        # other augmentations
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


class ClipLoadingSpectrogramPreprocessor(AudioToSpectrogramPreprocessor):
    """load audio samples from long audio files

    Directly loads a part of an audio file, eg 5-10 seconds, without loading
    entire file. This alows for prediction on long audio files without needing to
    pre-split or load large files into memory.

    It will load the requested audio segments into samples, regardless of length

    Args:
        df: a dataframe with file paths as index and 2 columns:
            ['start_time','end_time'] (seconds since beginning of file)
    Returns:
        ClipLoadingSpectrogramPreprocessor object

    Examples:
    You can quickly create such a df for a set of audio files like this:

    ```
    import librosa
    from opensoundscape.helpers import generate_clip_times_df
    files = glob('/path_to/*/*.WAV') #get list of full-length files
    clip_dfs = []
    clip_duration=5.0
    clip_overlap = 0.0
    for f in files:
        t = librosa.get_duration(filename=f)
        clips = generate_clip_times_df(t,clip_duration,clip_overlap)
        clips.index = [f]*len(clips)
        clips.index.name = 'file'
        clip_dfs.append(clips)
    clip_df = pd.concat(clip_dfs) #contains clip times for all files
    ```

    If you use this preprocessor with model.predict(), it will work, but
    the scores/predictions df will only have file paths not the times of clips.
    You will want to re-add the start and end times of clips as multi-index:

    ```
    score_df = model.predict(clip_loading_ds) #for instance
    score_df.index = pd.MultiIndex.from_arrays(
        [clip_df.index,clip_df['start_time'],clip_df['end_time']]
    )
    ```
    """

    def __init__(self, df):
        assert df.columns[0] == "start_time"
        assert df.columns[1] == "end_time"
        super(ClipLoadingSpectrogramPreprocessor, self).__init__(
            df, return_labels=False
        )

        self.actions.load_audio = actions.AudioClipLoader(sample_rate=None)
        self.pipeline[0] = self.actions.load_audio
        self.verbose = False
        self.specifies_clip_times = True

    def __getitem__(self, item_idx):

        try:
            df_row = self.df.iloc[item_idx]
            x = Path(df_row.name)  # the index contains a path to a file

            # perform each action in the pipeline if action.bypass==False
            for action in self.pipeline:
                if action.bypass:
                    continue
                elif (
                    hasattr(action, "requires_clip_times")
                    and action.requires_clip_times
                ):
                    x = action.go(x, df_row["start_time"], df_row["end_time"])
                elif action.requires_labels:
                    x, df_row = action.go(x, copy.deepcopy(df_row))
                else:
                    x = action.go(x)

            # this preprocessor does not support labels!
            #             # Return sample & label pairs (training/validation)
            #             if self.return_labels:
            #                 labels = torch.from_numpy(df_row.values)
            #                 return {"X": x, "y": labels}

            # Return sample only (prediction)
            return {"X": x}
        except:
            raise PreprocessingError(
                f"failed to preprocess sample: {self.df.index[item_idx]}"
            )
