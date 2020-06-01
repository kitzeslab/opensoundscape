from opensoundscape.utils.db_utils import write_ini_section


class OpensoundscapeModelFitAlgorithmDoesntExist(Exception):
    """
    model_fit algorithm doesn't exist exception
    """

    pass


def model_fit(config):
    """Fit a model

    Given a directory and method (from config), fit a model against the
    training data. This is an abstraction layer from the actual fitting.
    Essentially, we are going to look in `modules/model_fit_algo` for
    a directory named config['model_fit_algo'] and try to import a function
    called `model_fit_algo` from `model_fit_algo.py`

    Args:
        config: The parsed ini file for this run

    Returns:
        Something or possibly writes to MongoDB?

    Raises:
        Nothing
    """

    if config["model_fit"]["algo"] == "template_matching":
        from opensoundscape.model_fit.model_fit_algo.template_matching.model_fit_algo import (
            model_fit_algo,
        )
    else:
        raise OpensoundscapeModelFitAlgorithmDoesntExist(
            f"The algorithm '{config['model_fit']['algo']}' doesn't exist?"
        )

    if config["general"].getboolean("db_rw"):
        write_ini_section(config, "model_fit")

    model_fit_algo(config)
