import sys

def model_fit(config):
    '''Fit a model

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
    '''

    sys.path.append("modules/model_fit_algo/{}".format(config['model_fit']['algo']))
    from model_fit_algo import model_fit_algo

    model_fit_algo(config)
