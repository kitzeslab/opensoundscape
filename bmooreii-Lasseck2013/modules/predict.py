import sys

def predict(config):
    '''Make a prediction

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
    '''

    sys.path.append("modules/predict_algo/{}".format(config['predict']['algo']))
    from predict_algo import predict_algo

    predict_algo(config)
