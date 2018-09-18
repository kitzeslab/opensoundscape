import sys
from modules.db_utils import init_client
from modules.db_utils import close_client
from modules.db_utils import write_ini_section

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

    opensoundscape_dir = sys.path[0]
    sys.path.append(f"{opensoundscape_dir}/modules/spect_gen_algo/{config['spect_gen']['algo']}")
    from spect_gen_algo import spect_gen_algo

    if config['general'].getboolean('db_rw'):
        write_ini_section(config, 'spect_gen')

    spect_gen_algo(config)
