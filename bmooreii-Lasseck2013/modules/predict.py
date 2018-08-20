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

    openbird_dir = sys.path[0]
    sys.path.append(f"{openbird_dir}/modules/predict_algo/{config['predict']['algo']}")
    from predict_algo import predict_algo

    if config['general'].getboolean('db_rw'):
        write_ini_section(config, 'predict')

    predict_algo(config)
