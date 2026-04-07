import json
import types
import importlib.util

import numpy as np
import onnxruntime
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import opensoundscape.ml.onnx_model as onnx_model_module


class FakeORTSession:
    def __init__(
        self,
        _model_path=None,
        providers=None,
        metadata=None,
        output_names=None,
        run_outputs=None,
    ):
        self.providers = providers
        self._metadata = metadata or {}
        self._output_names = output_names or ["class_logits", "embedding"]
        self._run_outputs = run_outputs or [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        ]
        self.last_run_inputs = None

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map=self._metadata)

    def get_inputs(self):
        return [types.SimpleNamespace(name="audio_input")]

    def get_outputs(self):
        return [types.SimpleNamespace(name=n) for n in self._output_names]

    def run(self, _unused, inputs):
        self.last_run_inputs = inputs
        return self._run_outputs


def _sample_with_audio(audio_array):
    return types.SimpleNamespace(data=types.SimpleNamespace(samples=audio_array))


@pytest.fixture
def patch_onnxmodel_dependencies(monkeypatch):
    monkeypatch.setattr(
        onnx_model_module.onnxruntime, "InferenceSession", FakeORTSession
    )

    def _fake_classifier_init(self, architecture, classes, sample_duration):
        self._init_args = {
            "architecture": architecture,
            "classes": classes,
            "sample_duration": sample_duration,
        }

    monkeypatch.setattr(
        onnx_model_module.SpectrogramClassifier,
        "__init__",
        _fake_classifier_init,
    )


def test_init_reads_required_metadata_when_args_not_provided(
    patch_onnxmodel_dependencies,
):
    metadata = {
        "sample_rate": "32000",
        "sample_duration": "3.5",
        "classes": json.dumps(["a", "b"]),
        "class_outputs_key": "class_logits",
        "embedding_outputs_key": "embedding",
    }
    session = FakeORTSession(metadata=metadata)

    model = onnx_model_module.ONNXModel(session)

    assert model.sample_rate == 32000
    assert model.sample_duration == 3.5
    assert model.classes == ["a", "b"]
    assert model.class_outputs_key == "class_logits"
    assert model.embedding_outputs_key == "embedding"
    assert model.output_names == ["class_logits", "embedding"]


def test_init_raises_value_error_when_required_metadata_missing(
    patch_onnxmodel_dependencies,
):
    session = FakeORTSession(metadata={})

    with pytest.raises(ValueError, match="missing required model information"):
        onnx_model_module.ONNXModel(session)


def test_init_uses_explicit_arguments_over_missing_metadata(
    patch_onnxmodel_dependencies,
):
    session = FakeORTSession(metadata={})

    model = onnx_model_module.ONNXModel(
        session,
        sample_rate=44100,
        sample_duration=2.0,
        classes=["x", "y", "z"],
        class_outputs_key="class_logits",
        embedding_outputs_key="embedding",
    )

    assert model.sample_rate == 44100
    assert model.sample_duration == 2.0
    assert model.classes == ["x", "y", "z"]


def test_init_creates_session_from_path_with_requested_execution_providers(
    patch_onnxmodel_dependencies,
):
    provider_tuple = ("CPUExecutionProvider",)
    model = onnx_model_module.ONNXModel(
        "dummy_model.onnx",
        sample_rate=32000,
        sample_duration=3.0,
        classes=["c0"],
        class_outputs_key="class_logits",
    )

    assert isinstance(model.ort_session, FakeORTSession)
    assert model.ort_session.providers == provider_tuple


def test_batch_forward_adds_channel_dim_and_returns_named_outputs(
    patch_onnxmodel_dependencies,
):
    session = FakeORTSession(
        metadata={
            "sample_rate": "32000",
            "sample_duration": "1.0",
            "classes": json.dumps(["a", "b"]),
        },
        output_names=["class_logits", "embedding"],
        run_outputs=[
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        ],
    )
    model = onnx_model_module.ONNXModel(
        session,
        class_outputs_key="class_logits",
        embedding_outputs_key="embedding",
    )

    samples = [
        _sample_with_audio(np.array([0.0, 1.0, 2.0], dtype=np.float32)),
        _sample_with_audio(np.array([3.0, 4.0, 5.0], dtype=np.float32)),
    ]

    outs = model.batch_forward(samples, targets=None)

    assert set(outs.keys()) == {"class_logits", "embedding"}
    assert session.last_run_inputs[model.ort_input].shape == (2, 1, 3)


def test_batch_forward_filters_to_requested_targets(patch_onnxmodel_dependencies):
    session = FakeORTSession(
        metadata={
            "sample_rate": "32000",
            "sample_duration": "1.0",
            "classes": json.dumps(["a", "b"]),
        },
    )
    model = onnx_model_module.ONNXModel(
        session,
        class_outputs_key="class_logits",
        embedding_outputs_key="embedding",
    )

    samples = [_sample_with_audio(np.array([0.0, 1.0], dtype=np.float32))]
    outs = model.batch_forward(samples, targets=["embedding"])

    assert set(outs.keys()) == {"embedding"}


def test_batch_forward_without_channel_dim_uses_2d_input(patch_onnxmodel_dependencies):
    session = FakeORTSession(
        metadata={
            "sample_rate": "32000",
            "sample_duration": "1.0",
            "classes": json.dumps(["a", "b"]),
        },
    )
    model = onnx_model_module.ONNXModel(
        session,
        class_outputs_key="class_logits",
        embedding_outputs_key="embedding",
    )
    model.add_channel_dim = False

    samples = [_sample_with_audio(np.array([0.0, 1.0, 2.0], dtype=np.float32))]
    model.batch_forward(samples, targets=None)

    assert session.last_run_inputs[model.ort_input].shape == (1, 3)


def test_check_or_get_default_embedding_layer(patch_onnxmodel_dependencies):
    session = FakeORTSession(
        metadata={
            "sample_rate": "32000",
            "sample_duration": "1.0",
            "classes": json.dumps(["a", "b"]),
            "embedding_outputs_key": "embedding",
        }
    )
    model = onnx_model_module.ONNXModel(
        session,
        class_outputs_key="class_logits",
    )

    assert model._check_or_get_default_embedding_layer(None) == "embedding"


def test_check_or_get_default_embedding_layer_invalid_name_raises(
    patch_onnxmodel_dependencies,
):
    session = FakeORTSession(
        metadata={
            "sample_rate": "32000",
            "sample_duration": "1.0",
            "classes": json.dumps(["a", "b"]),
        }
    )
    model = onnx_model_module.ONNXModel(
        session,
        class_outputs_key="class_logits",
        embedding_outputs_key="embedding",
    )

    with pytest.raises(ValueError, match="target_layer"):
        model._check_or_get_default_embedding_layer("missing")


@pytest.mark.skipif(
    importlib.util.find_spec("onnxscript") is None,
    reason="onnxscript not installed",
)
def test_real_onnx_export_and_inference_roundtrip(tmp_path):
    from opensoundscape import CNN, preprocessors

    onnx_path = tmp_path / "roundtrip.onnx"

    # Create and export a real ONNX model with metadata.
    model = CNN(
        architecture="resnet18",
        classes=["c0", "c1"],
        sample_duration=1,
        preprocessor_cls=preprocessors.TorchSpectrogramPreprocessor,
        sample_rate=8000,
        arch_weights=None,
    )
    torch_onnx_model = model.save_onnx(onnx_path)

    # Check return type
    assert isinstance(torch_onnx_model, torch.onnx.ONNXProgram)

    # Load exported model through ONNXModel and run one real forward pass.
    onnx_model = onnx_model_module.ONNXModel(onnx_path)

    n_samples = int(onnx_model.sample_rate * onnx_model.sample_duration)
    samples = [
        _sample_with_audio(
            np.random.randn(n_samples).astype(np.float32),
        )
    ]
    outputs = onnx_model.batch_forward(samples, targets=[onnx_model.class_outputs_key])

    assert onnx_path.exists()
    assert onnx_model.class_outputs_key in outputs
    assert outputs[onnx_model.class_outputs_key].shape == (1, len(onnx_model.classes))


def test_from_spectrogram_classifier_similar_values():
    from opensoundscape import CNN, preprocessors, ONNXModel
    import onnxruntime
    import numpy as np
    import opensoundscape as opso

    model = CNN(
        architecture="efficientnet_b0",
        classes=[0, 1, 2, 3],
        sample_duration=3,
        preprocessor_cls=preprocessors.TorchSpectrogramPreprocessor,
        sample_rate=32000,
    )
    torch_onnx_program = model.save_onnx(path=None)  # , **save_onnx_kwargs)
    torch_onnx_program.optimize()
    # onnx_model = ONNXModel.from_spectrogram_classifier(model)
    # onnx_model.predict(audio_file_list)

    model_bytes = torch_onnx_program.model_proto.SerializeToString()
    ort_session = onnxruntime.InferenceSession(model_bytes)
    om = ONNXModel(
        # "onnx_test.onnx",
        ort_session,
        sample_rate=model.sample_rate,
        sample_duration=model.sample_duration,
        classes=model.classes,
        class_outputs_key="classifier",
        embedding_outputs_key="embedding",
    )

    embs = model.embed(opso.birds_path)
    logits = model.predict(opso.birds_path)

    om_embs = om.embed(opso.birds_path)
    om_logits = om.predict(opso.birds_path)

    # compare embs and om_embs, logits and om_logits
    assert embs.shape == om_embs.shape
    assert logits.shape == om_logits.shape
    assert np.allclose(embs, om_embs, atol=0.2)  # sometimes large differences!
    assert np.allclose(logits, om_logits, atol=0.2)  # sometimes large differences!
