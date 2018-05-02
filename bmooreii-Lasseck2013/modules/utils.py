from multiprocessing import cpu_count

def return_cpu_count(config):
    '''Return the number of CPUs requested

    If num_processors defined in the config, return that number. else return
    the number of CPUs on the machine.

    Args:
        config: The parsed ini file for this run

    Returns:
        nprocs: Integer number of cores
    '''
    if config['num_processors'] == '':
        nprocs = cpu_count()
    else:
        nprocs = config.getint('num_processors')
    return nprocs
