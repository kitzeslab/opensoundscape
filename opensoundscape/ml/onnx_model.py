from opensoundscape.ml.cnn import BaseModule
import onnxruntime as ort
import onnx
import numpy as np
from tqdm.autonotebook import tqdm
from opensoundscape.preprocess.preprocessors import AudioPreprocessor
import pandas as pd
from opensoundscape.ml.cnn import SpectrogramClassifier, _warn_output_size
from opensoundscape.ml.utils import apply_activation_layer
from opensoundscape.utils import identity


class ONNXModel(SpectrogramClassifier):
    """A model class for running ONNX models.

    The model is treated as frozen or immutable. If you wish to train a classifier on top of an
    ONNX model, you should extract embeddings using the embed() method and train
    opensoundscape.ml.shallow_classifier.MLPClassifier separately.

    This class wraps an ONNX model and provides a consistent interface
    with other model classes in OpenSoundscape.

    Attributes:
        onnx_session: The ONNX runtime session for the model.
    """

    def __init__(
        self,
        onnx_model_path,
        audio_sample_rate=None,
        sample_duration=None,
        classes=None,
        class_outputs_key=None,
        embedding_outputs_key=None,
        execution_providers=("CPUExecutionProvider",),
    ):
        """
        Initialize the ONNXModel from a model file.
        Args:
            onnx_model_path: Path to ONNX model file, or an ORT InferenceSession object
            audio_sample_rate: Sample rate model expects for input audio signals, Hz
                - if None, attempts to read from model metadata 'sample_rate'
            sample_duration: Duration model expects for input audio signals, seconds
                - if None, attempts to read from model metadata 'sample_duration'
            classes: List of class names corresponding to model outputs
                - if None, attempts to read from model metadata 'classes'
            class_outputs_key: Key to identify class output in model outputs
                - if None, attempts to read from model metadata 'class_outputs_key'
            embedding_outputs_key: Key to identify embedding output in model outputs
                - if None, attempts to read from model metadata 'embedding_outputs_key'
            execution_providers: Tuple of ONNX Runtime execution providers to use
                Including: "CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider", etc.
        """

        if isinstance(onnx_model_path, ort.InferenceSession):
            ort_session = onnx_model_path
        else:
            ort_session = ort.InferenceSession(
                str(onnx_model_path), providers=execution_providers
            )

        # audio_sample_rate, sample_duration, class_list, class_outputs_key are
        # either extracted from the ONNX model metadata or passed directly
        meta = ort_session.get_modelmeta().custom_metadata_map
        missing_fields = []
        if audio_sample_rate is None:
            try:
                audio_sample_rate = int(meta["sample_rate"])
            except Exception:
                missing_fields.append("audio_sample_rate")
        if sample_duration is None:
            try:
                sample_duration = float(meta["sample_duration"])
            except Exception:
                missing_fields.append("sample_duration")
        if classes is None:
            try:
                import json

                classes = json.loads(meta["classes"])
            except Exception:
                missing_fields.append("classes")
        if class_outputs_key is None:
            try:
                class_outputs_key = meta["class_outputs_key"]
            except Exception:
                pass
        if embedding_outputs_key is None:
            try:
                embedding_outputs_key = meta["embedding_outputs_key"]
            except Exception:
                pass
        if len(missing_fields) > 0:
            raise ValueError(
                f"ONNX model metadata is missing required model information: {missing_fields}. "
                "Please provide these arguments when initializing ONNXModel."
            )

        super().__init__(
            architecture="resnet18", classes=classes, sample_duration=sample_duration
        )  # TODO refactor so arch not required
        self.ort_session = ort_session
        self.ort_input = self.ort_session.get_inputs()[0].name
        self.audio_sample_rate = audio_sample_rate
        self.sample_duration = sample_duration
        self.classes = classes
        self.class_outputs_key = class_outputs_key
        self.embedding_outputs_key = embedding_outputs_key
        # should we require information on size of embedding output?
        self.embedding_dim = None
        self.add_channel_dim = True

        # get output names from the model
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

        # TODO: should we support multi-channel input? this could be configurable and
        # is probably good for future-proofing
        # self.include_channel_dim = False
        # self.n_audio_channels = 2

        self.preprocessor = AudioPreprocessor(
            sample_rate=self.audio_sample_rate,
            sample_duration=self.sample_duration,
        )

    def batch_forward(self, batch_samples, targets=(-1), avgpool=False):
        """Run inference on a batch of AudioSample

        Note: avgpool is currently not implemented for ONNX models, is ignored.

        Args:
            batch_samples: list of AudioSample objects
            targets: list of output keys to return from model outputs. If None, returns all outputs.
                e.g. ["class_logits", "embedding"]
        Returns:
            dict of output_key: np.ndarrays, one per requested output key
            each array is shape (batch_size, ...), where ... depends on model output shape
            - if targets is specified, only those outputs are returned
            - if targets is None, all model outputs are returned
        """

        # get audio samples from list of Audio objects, optionally add channel dimension
        if self.add_channel_dim:
            X = np.stack([s.data.samples[np.newaxis, :] for s in batch_samples])
        else:
            X = np.stack([s.data.samples for s in batch_samples])

        # run forward pass of ONNX model
        ort_outs = self.ort_session.run(None, {self.ort_input: X})

        # convert outputs to dict with named keys
        keys_subset = self.output_names if targets is None else targets
        return {
            name: ort_outs[i]
            for i, name in enumerate(self.output_names)
            if name in keys_subset
        }

    def _check_or_get_default_embedding_layer(self, target_layer):
        """Check or get default embedding layer

        Args:
            target_layer: requested target layer for embeddings
                - if None, uses self.embedding_outputs_key
        Returns:
            target_layer to use for embeddings
        """
        if target_layer is None:
            target_layer = self.embedding_outputs_key
        else:
            assert (
                target_layer in self.output_names
            ), f"Requested target_layer {target_layer} not found in model outputs: {self.output_names}"
        return target_layer

    # def __call__(
    #     self,
    #     dataloader,
    #     output_keys=None,
    #     progress_bar=True,
    # ):
    #     """Run inference on a dataloader, generating all outputs for each sample

    #     Args:
    #         dataloader: DataLoader object to create samples, e.g. from .predict_dataloader()
    #             Note: expects list of AudioSample objects, not (tensors, labels)
    #             This means you should use SafeAudioDataloader(...,collate_fn=identity)
    #         output_keys: list of output keys to return from model outputs. If None, returns all outputs.
    #             e.g. ["class_logits", "embedding"]
    #         progress_bar: bool, if True, shows a progress bar with tqdm [default: True]

    #     Returns:
    #         dict of output_key: np.ndarrays, one per requested output key
    #         each array is shape (num_samples, ...), where ... depends on model output shape
    #         - if output_keys is specified, only those outputs are returned
    #         - if output_keys is None, all model outputs are returned
    #     """
    #     if output_keys is not None:
    #         not_found = set(output_keys) - set(self.output_names)
    #         if len(not_found) > 0:
    #             raise ValueError(
    #                 f"Requested output_keys not found in model outputs: {not_found}. Model outputs: {self.output_names}"
    #             )

    #     if output_keys is None:  # return all outputs
    #         output_keys = self.output_names
    #     batch_outputs = []
    #     for batch in tqdm(dataloader, disable=not progress_bar):
    #         # get audio samples from list of Audio objects, and add channel dimension
    #         X = np.stack([s.data.samples[np.newaxis, :] for s in batch])

    #         # run forward pass of ONNX model
    #         ort_outs = self.ort_session.run(None, {self.ort_input: X})

    #         # convert outputs to dict with named keys
    #         keys_subset = self.output_names if output_keys is None else output_keys
    #         outputs = {
    #             name: ort_outs[i]
    #             for i, name in enumerate(self.output_names)
    #             if name in keys_subset
    #         }
    #         batch_outputs.append(outputs)
    #     if len(batch_outputs) == 0:
    #         return {k: [] for k in output_keys}
    #     # collate batch outputs into single dict of arrays
    #     return {
    #         key: np.concatenate(
    #             [batch_output[key] for batch_output in batch_outputs], axis=0
    #         )
    #         for key in output_keys
    #     }

    # def forward(
    #     self,
    #     samples,
    #     batch_size=1,
    #     num_workers=0,
    #     output_keys=None,
    #     progress_bar=True,
    #     return_dfs=False,
    #     audio_root=None,
    #     output_size_warning=1e9,
    #     **dataloader_kwargs,
    # ):
    #     """
    #     Generate model outputs for audio files/clips

    #     Wraps the creation of a dataloader and running inference to get outputs,
    #     and formats outputs as a dictionary of pd.DataFrames or np.ndarrays.

    #     Note: if return_dfs=True, all outputs must be 1 dimensional per sample

    #     Args:
    #         samples: same as .predict(): file path, list of file paths, OR pd.DataFrame with index
    #             containing audio file paths, OR a pd.DataFrame with multi-index (file, start_time,
    #             end_time)
    #         batch_size: batch size to use for dataloader [default: 1]
    #         num_workers: number of parallel CPU workers to use for dataloader [default: 0]
    #         output_keys: list or tuple of output keys to return from model outputs.
    #             See all available output keys in self.output_names.
    #             e.g. ("embedding",) or ("embedding","class_logits")
    #         progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
    #         avgpool: bool, if True, applies global average pooling to intermediate outputs
    #             i.e. averages across all dimensions except first to get a 1D vector per sample
    #         return_dfs: bool, if True, returns embeddings as pd.DataFrame with multi-index like
    #             .predict(). if False, returns np.array of embeddings [default: True]. If
    #             avg_pool=False, overrides to return np.array since we can't have a df with >2
    #             dimensions
    #         audio_root: optionally pass a root directory (pathlib.Path or str)
    #             - `audio_root` is prepended to each file path
    #             - if None (default), samples must contain full paths to files
    #         dataloader_kwargs are passed to self.predict_dataloader()

    #     Returns:
    #         If return_dfs=True:
    #             tuple of pd.DataFrame, each with multi-index (file, start_time, end_time)
    #             and columns corresponding to output dimensions (classes or embedding values)
    #         If return_dfs=False:
    #             tuple of np.ndarrays, each of shape (num_samples, ...), where ... is determined
    #             by model output shape for that output_key
    #     """
    #     if self.embedding_dim is not None:
    #         _warn_output_size(dataloader, self.embedding_dim, output_size_warning)

    #     # create a dataloader that loads batches of audio windows
    #     dataloader_kwargs.update(
    #         dict(batch_size=batch_size, num_workers=num_workers, audio_root=audio_root)
    #     )
    #     dataloader = self.predict_dataloader(
    #         samples,
    #         collate_fn=identity,
    #         **dataloader_kwargs,
    #     )

    #     # run the forward pass on each batch of the dataloader to get all outputs
    #     outputs = self(
    #         dataloader,
    #         output_keys=output_keys,
    #         progress_bar=progress_bar,
    #     )

    #     if return_dfs:  # place outputs in dataframes with clip info as index
    #         for key in output_keys:
    #             if len(outputs[key].shape) > 2:
    #                 continue  # leave as np.ndarray if >2 dimensions
    #             try:
    #                 outputs[key] = pd.DataFrame(
    #                     data=outputs[key],
    #                     index=dataloader.dataset.dataset.label_df.index,
    #                     columns=self.classes if key == self.class_outputs_key else None,
    #                 )
    #             except Exception as e:
    #                 # leave outputs as np.ndarray if DataFrame creation fails
    #                 pass
    #     return outputs

    # def embed(
    #     self,
    #     samples,
    #     batch_size=1,
    #     num_workers=0,
    #     output_key=None,
    #     progress_bar=True,
    #     return_dfs=True,
    #     audio_root=None,
    #     output_size_warning=1e9,
    #     **dataloader_kwargs,
    # ):
    #     """
    #     Generate embeddings for audio files/clips

    #     Returns the model outputs for output key (defaults to self.embedding_outputs_key).
    #     To get multiple outputs, use .forward() instead.

    #     Args:
    #         samples: same as .predict(): file path, list of file paths, OR pd.DataFrame with index
    #             containing audio file paths, OR a pd.DataFrame with multi-index (file, start_time,
    #             end_time)
    #         batch_size: batch size to use for dataloader [default: 1]
    #         num_workers: number of parallel CPU workers to use for dataloader [default: 0]
    #         output_key: Which model output to return as embeddings.
    #             See all available output keys in self.output_names.
    #             Default [None] selects the output corresponding to self.embedding_outputs_key.
    #         progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
    #         return_dfs: bool, if True, returns embeddings as pd.DataFrame with multi-index like
    #             .predict(). if False, returns np.array of embeddings [default: True]. If
    #             avg_pool=False, overrides to return np.array since we can't have a df with >2
    #             dimensions
    #         audio_root: optionally pass a root directory (pathlib.Path or str)
    #             - `audio_root` is prepended to each file path
    #             - if None (default), samples must contain full paths to files
    #         output_size_warning: if not None,
    #         dataloader_kwargs are passed to self.predict_dataloader()

    #     Returns:
    #         If return_dfs=True:
    #             pd.DataFrame with multi-index (file, start_time, end_time)
    #         If return_dfs=False:
    #             np.ndarray of shape (num_samples, ...embedding_shape...)
    #     """
    #     output_key = output_key or self.embedding_outputs_key
    #     if output_key is None:
    #         raise ValueError(
    #             "embedding_outputs_key is not set for this model. "
    #             "Please specify output_key of desired outputs."
    #         )
    #     return self.forward(
    #         samples,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         output_keys=(output_key,),
    #         progress_bar=progress_bar,
    #         return_dfs=return_dfs,
    #         audio_root=audio_root,
    #         output_size_warning=output_size_warning,
    #         **dataloader_kwargs,
    #     )[output_key]

    # def predict(
    #     self,
    #     samples,
    #     batch_size=1,
    #     num_workers=0,
    #     activation_layer=None,
    #     progress_bar=True,
    #     audio_root=None,
    #     output_size_warning=1e9,
    #     **dataloader_kwargs,
    # ):
    #     """Generate predictions on a set of samples

    #     Return dataframe of model output scores for each sample.
    #     Optional activation layer for scores
    #     (softmax, sigmoid, softmax then logit, or None)

    #     Args:
    #         samples:
    #             the files to generate predictions for. Can be:
    #             - a dataframe with index containing audio paths, OR
    #             - a dataframe with multi-index (file, start_time, end_time), OR
    #             - a list (or np.ndarray) of audio file paths
    #             - a single file path (str or pathlib.Path)
    #         batch_size:
    #             Number of files to load simultaneously [default: 1]
    #         num_workers:
    #             parallelization (ie cpus or cores), use 0 for current process
    #             [default: 0]
    #         activation_layer:
    #             Optionally apply an activation layer such as sigmoid or
    #             softmax to the raw outputs of the model.
    #             options:
    #             - None: no activation, return raw logit scores [-inf:inf]
    #             - 'softmax': scores all classes sum to 1, scores between 0 and 1
    #             - 'sigmoid': each class is independent, scores between 0 and 1
    #             - 'softmax_and_logit': applies softmax first then logit
    #             [default: None]
    #         split_files_into_clips:
    #             If True, internally splits and predicts on clips from longer audio files
    #             Otherwise, assumes each row of `samples` corresponds to one complete sample
    #         clip_overlap_fraction, clip_overlap, clip_step, final_clip:
    #             see `opensoundscape.utils.generate_clip_times_df`
    #         overlap_fraction: deprecated alias for clip_overlap_fraction
    #         bypass_augmentations: If False, Actions with
    #             is_augmentation==True are performed. Default True.
    #         invalid_samples_log: if not None, samples that failed to preprocess
    #             will be listed in this text file.
    #         raise_errors:
    #             if True, raise errors when preprocessing fails
    #             if False, just log the errors to unsafe_samples_log
    #         wandb_session: a wandb session to log to
    #             - pass the value returned by wandb.init() to progress log to a
    #             Weights and Biases run
    #             - if None, does not log to wandb
    #         return_invalid_samples: bool, if True, returns second argument, a set
    #             containing file paths of samples that caused errors during preprocessing
    #             [default: False]
    #         progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
    #         audio_root: optionally pass a root directory (pathlib.Path or str)
    #             - `audio_root` is prepended to each file path
    #             - if None (default), samples must contain full paths to files
    #         output_size_warning: int, if >0, raises a warning if the number of
    #             output scores (clips * classes) exceeds this number, as this
    #             can cause heavy memory usage. Set to None or 0 to disable.
    #             [default: 1e9]
    #         **dataloader_kwargs: additional arguments to self.predict_dataloader()
    #             such as:
    #                 clip_overlap=None,
    #                 clip_overlap_fraction=None,
    #                 clip_step=None,
    #                 final_clip="extend",
    #                 bypass_augmentations=True,
    #                 invalid_samples_log=None,
    #                 raise_errors=False,
    #     Returns:
    #         df of post-activation_layer scores
    #         - if return_invalid_samples is True, returns (df,invalid_samples)
    #         where invalid_samples is a set of file paths that failed to preprocess

    #     Effects:
    #         (1) unsafe sample logging
    #         If unsafe_samples_log is not None, saves a list of all file paths that
    #         failed to preprocess in unsafe_samples_log as a text file

    #     Note: if loading an audio file raises a PreprocessingError, the scores
    #         for that sample will be np.nan
    #     """
    #     logits = self.forward(
    #         samples,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         output_keys=(self.class_outputs_key,),
    #         progress_bar=progress_bar,
    #         audio_root=audio_root,
    #         output_size_warning=output_size_warning,
    #         return_dfs=True,  # we will convert to df after applying activation
    #         **dataloader_kwargs,
    #     )[self.class_outputs_key]
    #     scores = apply_activation_layer(logits.values, activation_layer)
    #     return pd.DataFrame(data=scores, index=logits.index, columns=logits.columns)

    # def embed_to_hoplite_db(self, samples, db, **kwargs):
    #     """can provide flag as to whether the embeddings or classifier outputs should be stored in db"""
    #     raise NotImplementedError()

    # def predict_to_hoplite_db(self, samples, db, **kwargs):
    #     """applies self.predict to samples and stores results in db"""
    #     raise NotImplementedError()

    # def similarity_search_hoplite_db(self, query_samples, db, **kwargs):
    #     raise NotImplementedError()
