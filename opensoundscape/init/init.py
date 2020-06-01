from opensoundscape.utils.db_utils import write_ini_section


def init(config):
    """Initialize INI DB


    Args:
        config: The parsed ini file for this run

    Returns:
        Writes to DB

    Raises:
        Nothing
    """

    if config["general"].getboolean("db_rw"):
        write_ini_section(config, "general")
