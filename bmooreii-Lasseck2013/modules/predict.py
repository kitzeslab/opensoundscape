import sys
from modules.db_utils import write_ini_section

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

    sys.path.append(f"modules/predict_algo/{config['predict']['algo']}")
    from predict_algo import predict_algo

    if config['general'].getboolean('db_rw'):
        write_ini_section(config, 'predict')

    predict_algo(config)
