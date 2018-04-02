import sys

def model_fit(dir, config):
    '''Fit a model

    Given a directory and method (from config), fit a model against the
    training data.

    Args:
        dir: A directory containing training data
        config: The parsed ini file for this run

    Returns:
        Nothing. Writes a model to the MongoDB.

    Raises:
        Nothing
    '''

    sys.path.append("modules/model_fit_algo/{}".format(config['model_fit_algo']))
    from model_fit_algo import model_fit_algo

    model_fit_algo(dir, config)
