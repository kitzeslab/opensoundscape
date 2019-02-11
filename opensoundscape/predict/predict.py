from opensoundscape.utils.db_utils import write_ini_section


class OpensoundscapePredictAlgorithmDoesntExist(Exception):
    """
    predict algorithm doesn't exist exception
    """

    pass


def predict(config):
    """Make a prediction

    Given a `predict_algo` method (from config), make a prediction based on
    that model.

    Args:
        config: The parsed ini file for this run

    Returns:
        A prediction

    Raises:
        Nothing

    To Do:
        -
    """

    if config["predict"]["algo"] == "lasseck2013":
        from opensoundscape.predict.predict_algo.lasseck2013.predict_algo import (
            predict_algo,
        )
    else:
        raise OpensoundscapePredictAlgorithmDoesntExist(
            f"The predict algorithm '{config['predict']['algo']}' doesn't exist"
        )

    if config["general"].getboolean("db_rw"):
        write_ini_section(config, "predict")

    predict_algo(config)
