"""lightweight wrapper to list and get models from bioacoustics model zoo with torch.hub"""
import torch


def list_models():
    """
    list the models available in the [bioacoustics model zoo](https://github.com/kitzeslab/bioacoustics-model-zoo)

    Returns:
        list of available models

    see also: load(model)
    """
    return torch.hub.list("kitzeslab/bioacoustics-model-zoo")


def load(model):
    """
    load a model from the [bioacoustics model zoo](https://github.com/kitzeslab/bioacoustics-model-zoo)

    see list_models() for a list of available models

    Args:
        model: name of model to load, i.e. one listed by list_models()

    Returns:
        ready-to-use model object
        - can typically be used just like CNN classs: model.predict()
        - see the model zoo page for details on each model

    note that some models may require additional arguments; in that case,
    use torch.hub.load("kitzeslab/bioacoustics-model-zoo", ...) directly,
    passing additional arguments after the model name
    (see https://github.com/kitzeslab/bioacoustics-model-zoo landing page for
    detailed instructions)

    """
    return torch.hub.load("kitzeslab/bioacoustics-model-zoo", model)
