import sys

def spect_gen(config):
    '''Fit a model

    Given a method (from config), generate spectrum for all input files.

    Args:
        config: The parsed ini file for this run

    Returns:
        Nothing or writes to MongoDB

    Raises:
        Nothing
    '''

    sys.path.append("modules/spect_gen_algo/{}".format(config['spect_gen']['algo']))
    from spect_gen_algo import spect_gen_algo

    spect_gen_algo(config)
