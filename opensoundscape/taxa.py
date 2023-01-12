"""a set of utilites for converting between scientific and common names of bird species in different
naming systems (xeno canto and bird net)"""
import importlib.resources

import numpy as np
import pandas as pd

with importlib.resources.path("opensoundscape.resources", "species_table.csv") as f:
    species_table = pd.read_csv(f)


def get_species_list():
    """returns a list of scientific-names (lowercase-hyphenated) of species"""

    return (
        species_table[["bn_code", "bn_mapping_in_xc_dataset"]]
        .dropna()["bn_mapping_in_xc_dataset"]
        .sort_values()
        .to_list()
    )


name_table_sci_idx = species_table.set_index("scientific", drop=True)
name_table_xc_com_idx = species_table.set_index("xc_common", drop=True)
name_table_bn_com_idx = species_table.set_index("bn_common", drop=True)


def sci_to_bn_common(scientific):
    """convert scientific lowercase-hyphenated name to birdnet common name as lowercasenospaces"""
    return name_table_sci_idx.at[scientific, "bn_common"]


def sci_to_xc_common(scientific):
    """convert scientific lowercase-hyphenated name to xeno-canto lowercasenospaces common name"""
    return name_table_sci_idx.at[scientific, "xc_common"]


def xc_common_to_sci(common):
    """convert xeno-canto common name (ignoring dashes/spaces/case) to lowercase-hyphenated name"""
    common = common.lower().replace(" ", "").replace("-", "")
    return name_table_xc_com_idx.at[common, "scientific"]


def bn_common_to_sci(common):
    """convert bird net common name (ignoring dashes, spaces, case) to lowercase-hyphenated name"""
    common = common.lower().replace(" ", "").replace("-", "")
    return name_table_bn_com_idx.at[common, "scientific"]


def common_to_sci(common):
    """convert bird net common name to scientific name as lowercase-hyphenated

    Ignores dashes, spaces, case
    """
    return bn_common_to_sci(common)
