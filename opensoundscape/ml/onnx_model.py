import json

import numpy as np
import torch

from opensoundscape.ml.cnn import SpectrogramClassifier
from opensoundscape.preprocess.preprocessors import AudioPreprocessor


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
        sample_rate=None,
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
            sample_rate: Sample rate model expects for input audio signals, Hz
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
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required to use ONNXModel. Please install with `pip install onnxruntime` "
                "or `pip install opensoundscape[onnx]`."
            )

        if isinstance(onnx_model_path, ort.InferenceSession):
            ort_session = onnx_model_path
        else:
            ort_session = ort.InferenceSession(
                str(onnx_model_path), providers=execution_providers
            )

        # sample_rate, sample_duration, class_list, class_outputs_key are
        # either extracted from the ONNX model metadata or passed directly
        meta = ort_session.get_modelmeta().custom_metadata_map
        missing_fields = []
        if sample_rate is None:
            try:
                sample_rate = int(meta["sample_rate"])
            except Exception:
                missing_fields.append("sample_rate")
        if sample_duration is None:
            try:
                sample_duration = float(meta["sample_duration"])
            except Exception:
                missing_fields.append("sample_duration")
        if classes is None:
            try:
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

        # the Torch Model (.network) of SpectrogramClassifier will not be used
        # pass an Identity just as a placeholder
        super().__init__(
            architecture=torch.nn.Identity(),  # not used, but required for super init
            classes=classes,
            sample_duration=sample_duration,
            sample_rate=sample_rate,
        )
        self.ort_session = ort_session
        self.ort_input = self.ort_session.get_inputs()[0].name
        self.classes = classes
        self.class_outputs_key = class_outputs_key
        self.embedding_outputs_key = embedding_outputs_key
        self.embedding_dim = None

        self.add_channel_dim = True
        """whether to add a channel dimension to the input audio (e.g. for mono models that expect shape (batch_size, 1, n_samples))
        if False, input audio will be shape (batch_size, n_samples)
        if True, input audio will be shape (batch_size, 1, n_samples)
        """

        # get output names from the model
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

        # TODO: should we support multi-channel input? this could be configurable
        # and is probably good for future-proofing
        # self.n_audio_channels = 2 # None for no channel dim, 1 to add channel dim to mono, 2 for stereo audio, etc

        self.preprocessor = AudioPreprocessor(
            sample_duration=sample_duration,
            sample_rate=sample_rate,
        )

    def batch_forward(self, batch_samples, targets=None, avgpool=False):
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
            if target_layer not in self.output_names:
                raise ValueError(
                    f"Requested target_layer {target_layer} not found in model outputs: {self.output_names}"
                )
        return target_layer

    @classmethod
    def from_spectrogram_classifier(
        cls,
        model: SpectrogramClassifier,
        execution_providers=("CPUExecutionProvider",),
        **save_onnx_kwargs,
    ):
        """Create an ONNXModel from a SpectrogramClassifier (pytorch model).

        This method exports the given PyTorch model to ONNX format and initializes an ONNXModel with it.
        Note that the resulting ONNXModel is frozen and cannot be further trained. If you wish to train
        a classifier on top of the ONNX model, you should extract embeddings using the embed() method and
        train opensoundscape.ml.shallow_classifier.MLPClassifier separately.

        Currently onnx export from SpectrogramClassifier requires that the preprocessor is a TorchSpectrogramPreprocessor

        Note: due to implementation details, ONNXmodels can produce slightly different outputs for classifier and embedding outputs,
        e.g. +/- 0.02 on logits and +/- 0.1 on embeddings in some experiments

        Args:
            model: A SpectrogramClassifier object to convert to ONNXModel.
            execution_providers: Tuple of ONNX Runtime execution providers to use for the resulting ONNXModel.
                 EG: "CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"
            **save_onnx_kwargs: Additional kwargs to pass to model.save_onnx() method:
                activation_layer: Any | None = None,
                include_preprocessor_output: bool = True,
                include_embedding_output: bool = True,
                include_classifier_output: bool = True,

        Returns:
            An ONNXModel initialized with the exported ONNX model.

        Example usage:
        ```python
        from opensoundscape import CNN, preprocessors, ONNXModel
        model = CNN(
            architecture="efficientnet_b0",
            classes=[0, 1, 2, 3],
            sample_duration=3,
            preprocessor_cls=preprocessors.TorchSpectrogramPreprocessor,
            sample_rate=32000,
        )
        onnx_model = ONNXModel.from_spectrogram_classifier(model)
        onnx_model.predict(audio_file_list)
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required to use ONNXModel. Please install with `pip install onnxruntime` "
                "or `pip install opensoundscape[onnx]`."
            )
        torch_onnx_program = model.save_onnx(path=None, **save_onnx_kwargs)
        torch_onnx_program.optimize()

        model_bytes = torch_onnx_program.model_proto.SerializeToString()
        ort_session = ort.InferenceSession(model_bytes, providers=execution_providers)
        return cls(
            ort_session,
            sample_rate=model.preprocessor.sample_rate,
            sample_duration=model.preprocessor.sample_duration,
            classes=model.classes,
            class_outputs_key="classifier",
            embedding_outputs_key="embedding",
            execution_providers=execution_providers,
        )
