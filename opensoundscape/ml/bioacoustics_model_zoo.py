"""lightweight wrapper to list and get models from bioacoustics model zoo with torch.hub"""

import torch


def list_models(**kwargs):
    """
    list the models available in the [bioacoustics model zoo](https://github.com/kitzeslab/bioacoustics-model-zoo)

    Args:
        **kwargs are passed to torch.hub.list()
    Returns:
        list of available models

    see also: load(model)
    """
    tag = "0.11.0.dev1"  # in the future, might be based on opensoundscape.__version__
    return torch.hub.list(
        f"kitzeslab/bioacoustics-model-zoo:{tag}", trust_repo=True, **kwargs
    )


def load(model, tag=None, **kwargs):
    """
    load a model from the [bioacoustics model zoo](https://github.com/kitzeslab/bioacoustics-model-zoo)

    see list_models() for a list of available models

    chooses the repository tag based on the opensoundscape version

    Args:
        model: name of model to load, i.e. one listed by list_models()
        tag: branch or tag of GitHub repository to use
            - if None, selects tag based on OpenSoundscape version
        **kwargs are passed to torch.hub.load()

    Returns:
        ready-to-use model object
        - can typically be used just like CNN classs: model.predict()
        - see the model zoo page for details on each model

    note that some models may require additional arguments; in that case,
    use torch.hub.load("kitzeslab/bioacoustics-model-zoo", ...) directly,
    passing additional arguments after the model name
    (see https://github.com/kitzeslab/bioacoustics-model-zoo landing page for
    detailed instructions)

    > Note on Error "module not found: bioacoustics_model_zoo" when using multiprocessing (num_workers>0):
    if you get an error to this effect, please install the bioacoustics_model_zoo as a package in your python environment:
    > `pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo@0.11.0.dev1`
    > as the torch.hub api seems to have trouble with multiprocessing for some model classes.

    """
    if tag is None:
        # in the future, might be based on opensoundscape.__version__
        tag = "0.11.0.dev1"
    return torch.hub.load(
        f"kitzeslab/bioacoustics-model-zoo:{tag}", model, trust_repo=True, **kwargs
    )
