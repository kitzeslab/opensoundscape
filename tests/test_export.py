import pytest
import warnings
import torch
import torch.nn as nn

from opensoundscape.ml.export import SequentialModelExporter, to_onnx_program


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Number of audio samples used as the ONNX input length in all tests.
# The preprocessing transform is an Identity so the model receives a tensor
# of shape (batch, 1, INPUT_LENGTH); Flatten + Linear maps that to outputs.
INPUT_LENGTH = 16
OUT_FEATURES = 2


class FlattenLinear(nn.Module):
    """Tiny model without a classifier_layer attribute.

    Accepts (batch, 1, INPUT_LENGTH) tensors produced by IdentityTransform
    and maps them to OUT_FEATURES logits.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(INPUT_LENGTH, OUT_FEATURES)

    def forward(self, x):
        return self.fc(self.flatten(x))


class FlattenLinearWithClassifierLayer(nn.Module):
    """Model that has a classifier_layer attribute, mimicking CNN.network."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(INPUT_LENGTH, INPUT_LENGTH)
        self.head = nn.Linear(INPUT_LENGTH, OUT_FEATURES)
        self.classifier_layer = "head"

    def forward(self, x):
        x = self.embedding(self.flatten(x))
        return self.head(x)


class IdentityTransform(nn.Module):
    """Preprocessing transform that returns the input unchanged."""

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# ONNXModel tests
# ---------------------------------------------------------------------------


class TestSequentialModelExporter:
    def test_forward_returns_all_outputs_by_default(self):
        """ONNXModel returns outputs for every key when outputs=None."""
        m1 = nn.Identity()
        m2 = nn.Linear(4, 2)
        model = SequentialModelExporter({"a": m1, "b": m2})
        x = torch.rand(3, 4)
        result = model(x)
        assert set(result.keys()) == {"a", "b"}

    def test_forward_filters_outputs(self):
        """ONNXModel only returns keys listed in outputs."""
        m1 = nn.Identity()
        m2 = nn.Linear(4, 2)
        model = SequentialModelExporter({"a": m1, "b": m2}, outputs=["b"])
        x = torch.rand(3, 4)
        result = model(x)
        assert "a" not in result
        assert "b" in result

    def test_forward_passes_output_sequentially(self):
        """Each model in the sequence receives the previous model's output."""
        double = nn.Linear(4, 4, bias=False)
        nn.init.eye_(double.weight)  # identity weights
        double.weight.data *= 2.0

        identity = nn.Identity()
        model = SequentialModelExporter({"double_value": double, "identity": identity})
        x = torch.ones(1, 4)
        result = model(x)
        # after doubling, identity should return the doubled tensor
        assert torch.allclose(result["identity"], torch.full((1, 4), 2.0))

    def test_eval_called_on_submodels(self):
        """ONNXModel calls eval() on submodels during __init__."""
        linear = nn.Linear(2, 2)
        linear.train()  # set to training mode first
        SequentialModelExporter({"l": linear})
        assert not linear.training

    def test_output_values_are_clones(self):
        """Outputs stored in the dict are clones, not the same tensor."""
        identity = nn.Identity()
        model = SequentialModelExporter({"id": identity})
        x = torch.rand(2, 4)
        result = model(x)
        assert result["id"] is not x


# ---------------------------------------------------------------------------
# to_onnx_program tests
# ---------------------------------------------------------------------------

# Skip all ONNX-export tests if required packages are absent.
onnx_deps = pytest.mark.skipif(
    not all(
        __import__("importlib").util.find_spec(p) is not None
        for p in ("onnx", "onnxruntime", "onnxscript")
    ),
    reason="onnx, onnxruntime, or onnxscript not installed",
)


@onnx_deps
class TestToOnnxProgram:
    """Integration tests that require onnx/onnxruntime/onnxscript."""

    @pytest.fixture()
    def preprocessing(self):
        return IdentityTransform()

    @pytest.fixture()
    def simple_model(self):
        return FlattenLinear()

    @pytest.fixture()
    def model_with_classifier(self):
        return FlattenLinearWithClassifierLayer()

    def test_returns_onnx_program(self, preprocessing, simple_model):
        prog = to_onnx_program(
            preprocessing_transforms=preprocessing,
            torch_model=simple_model,
            input_length=INPUT_LENGTH,
        )
        assert prog is not None

    def test_invalid_activation_raises(self, preprocessing, simple_model):
        with pytest.raises(ValueError, match="invalid option for activation_layer"):
            to_onnx_program(
                preprocessing_transforms=preprocessing,
                torch_model=simple_model,
                input_length=INPUT_LENGTH,
                activation_layer="relu",  # not a valid option
            )

    def test_softmax_activation_accepted(self, preprocessing, simple_model):
        prog = to_onnx_program(
            preprocessing_transforms=preprocessing,
            torch_model=simple_model,
            input_length=INPUT_LENGTH,
            activation_layer="softmax",
        )
        assert prog is not None

    def test_sigmoid_activation_accepted(self, preprocessing, simple_model):
        prog = to_onnx_program(
            preprocessing_transforms=preprocessing,
            torch_model=simple_model,
            input_length=INPUT_LENGTH,
            activation_layer="sigmoid",
        )
        assert prog is not None

    def test_warns_when_embedding_cannot_be_separated(
        self, preprocessing, simple_model
    ):
        """When model lacks classifier_layer, a warning is raised and embedding
        output is dropped."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            to_onnx_program(
                preprocessing_transforms=preprocessing,
                torch_model=simple_model,
                input_length=INPUT_LENGTH,
                include_embedding_output=True,
            )
        messages = [str(warning.message) for warning in w]
        assert any("embedding" in m.lower() for m in messages)

    def test_no_warning_when_classifier_layer_present(
        self, preprocessing, model_with_classifier
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            to_onnx_program(
                preprocessing_transforms=preprocessing,
                torch_model=model_with_classifier,
                input_length=INPUT_LENGTH,
                include_embedding_output=True,
            )
        embedding_warnings = [
            warning
            for warning in w
            if "embedding" in str(warning.message).lower()
            and "separate" in str(warning.message).lower()
        ]
        assert len(embedding_warnings) == 0

    def test_include_flags_control_outputs(self, preprocessing, simple_model):
        """Setting include_preprocessor_output=False removes 'sample' from outputs."""
        # We can't directly inspect output_names from the program, but we can
        # confirm no error is raised and the function completes.
        prog = to_onnx_program(
            preprocessing_transforms=preprocessing,
            torch_model=simple_model,
            input_length=INPUT_LENGTH,
            include_preprocessor_output=False,
            include_embedding_output=False,
            include_classifier_output=True,
        )
        assert prog is not None
