from opensoundscape.ml.cnn_architectures import set_layer_from_name
import copy
import warnings
import torch


class ONNXModel(torch.nn.Module):
    def __init__(self, sequential_models, outputs=None):
        super(ONNXModel, self).__init__()

        for name, m in sequential_models.items():
            try:
                m.eval()
            except:
                pass

        self.sequential_models = sequential_models

        # determine which of the sequential model outputs to include in output dict
        if outputs is None:
            # retain outputs of each sequential model by default
            outputs = list(sequential_models.keys())
        self.outputs = outputs

    def forward(self, x):
        outs = {}
        for name, m in self.sequential_models.items():
            x = m(x)
            if name in self.outputs:
                # only include the outputs if listed in self.outputs
                outs[name] = x.clone()
        return outs


def to_onnx_program(
    preprocessing_transforms,
    torch_model,
    input_length,
    activation_layer=None,
    include_preprocessor_output=True,
    include_embedding_output=True,
    include_classifier_output=True,
    opset_version=18,
    **kwargs,
):
    """Export a torch model with preprocessing transforms to ONNX format

    Attempts to separate embedding and classifier outputs from torch_model, if
    torch_model has attribute 'classifier_layer' indicating the name of the layer
    that should be considered the "classifier". The remaining layers are considered
    the "embedding" portion of the network. There should be no layers after the
    classifier layer.

    Optionally adds a sigmoid or softmax activation layer on the classifier outputs.

    Requires that onnx, onnxruntime, and onnxscript are packages are installed

    Args:
        preprocessing_transforms: torch.nn.Module, preprocessing transforms to apply to raw audio
        torch_model: torch.nn.Module, model to export
        input_length: int, length of input audio samples in number of samples
        activation_layer: str or None, activation layer to apply to classifier outputs
            options: None, 'softmax', 'sigmoid'
        include_preprocessor_output: bool, whether to include preprocessor output in ONNX model outputs
        include_embedding_output: bool, whether to include embedding output in ONNX model outputs
        include_classifier_output: bool, whether to include classifier output in ONNX model outputs
        opset_version: int, ONNX opset version to use for export
            currently defaults to 18 because of issues with dynamic shapes in 20 with pytorch 2.9.0;
            should upgrade to 20 when stable fixes are released
        **kwargs: additional keyword arguments to pass to torch.onnx.export
    Returns:
        onnx_model: ONNX program model object

    Example:
    ```python
    from opensoundscape import Audio, Spectrogram, CNN, BoxedAnnotations, preprocessors

    model = CNN(
        architecture="efficientnet_b0",
        classes=[0, 1, 2, 3],
        sample_duration=3,
        preprocessor_cls=preprocessors.TorchSpectrogramPreprocessor,
        sample_rate=32000,
    )
    # a list of torchaudio preprocesesing transforms such as Spectrogram, MelSpectrogram, etc.
    transforms=model.preprocessor["transforms"].transforms

    # expected number of samples in input audio: 3*32000
    input_length = model.preprocessor.sample_rate * model.preprocessor.sample_duration

    onnx_program = to_onnx_program(
        preprocessing_transforms=transforms,
        torch_model=model.network,
        input_length=input_length,
        include_preprocessor_output=True,
    )
    onnx_program.save("efficientnet_b0.onnx")
    ```
    """
    # check for dependency packages
    try:
        import onnx, onnxruntime, onnxscript
    except ImportError:
        warnings.warn(
            "to_onnx_program may fail since at least one of onnx, onnxruntime, and onnxscript packages are not installed"
        )
    # attempt to separate embeddings and classifier outputs
    outputs = []
    if include_preprocessor_output:
        outputs.append("sample")
    if include_embedding_output:
        outputs.append("embedding")
    if include_classifier_output:
        outputs.append("classifier")

    # attempt to separate embeddings and classifier layers, based on the
    # torch_model.network.classifier_layer attribute
    try:
        assert hasattr(torch_model, "classifier_layer")
        embedding_model = copy.deepcopy(torch_model)
        set_layer_from_name(
            embedding_model, embedding_model.classifier_layer, torch.nn.Identity()
        )
        classifier = torch.nn.Module.get_submodule(
            torch_model, torch_model.classifier_layer
        )
    except:
        classifier = torch_model
        if include_embedding_output:
            outputs.remove("embedding")
            warnings.warn(
                """Could not separate embedding and classifier outputs; exporting
                model without embedding outputs. To separate, make sure
                torch_model has attribute 'classifier_layer' indicating the name
                of the layer to separate embeddings from classifier."""
            )

    # add the optional activation layer to the classifier
    if activation_layer is None:
        pass
    elif activation_layer == "softmax":
        activation_module = torch.nn.Softmax(dim=1)
        classifier = torch.nn.Sequential(classifier, activation_module)
    elif activation_layer == "sigmoid":
        activation_module = torch.nn.Sigmoid()
        classifier = torch.nn.Sequential(classifier, activation_module)
    else:
        raise ValueError(f"invalid option for activation_layer: {activation_layer}")

    if "embedding" in outputs:
        onnx_model = ONNXModel(
            {
                "sample": preprocessing_transforms,
                "embedding": embedding_model,
                "classifier": classifier,
            }
        )
    else:
        onnx_model = ONNXModel(
            {
                "sample": preprocessing_transforms,
                "classifier": classifier,
            }
        )

    # create a sample input tensor for ONNX export
    # the input size to the model will include a channel dimension of size 1
    # we pass an example input with batch size 2 for ONNX export
    # the resulting model allows dynamic batch size
    example_input_batch = torch.rand(2, 1, input_length)

    return torch.onnx.export(
        onnx_model,
        (example_input_batch,),
        dynamic_shapes=[{0: "dim_x"}],  # allow dynamic batch size
        dynamo=True,
        output_names=outputs,
        opset_version=opset_version,
        **kwargs,  # e.g. report=True
    )
