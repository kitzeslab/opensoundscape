import numpy as np
import onnxruntime as ort

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

    def batch_forward(self, batch_samples, targets=(-1,), avgpool=False):
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
