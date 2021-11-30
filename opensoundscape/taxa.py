"""a set of utilites for converting between scientific and common names of bird species in different naming systems (xeno canto and bird net)"""
import numpy as np
import pandas as pd
import importlib.resources

with importlib.resources.path("opensoundscape.resources", "species_table.csv") as f:
    species_table = pd.read_csv(f)


def get_species_list():
    """list of scientific-names (lowercase-hyphenated) of species in the loaded species table"""

    # create a dictionary that maps from 6 letter bn codes to xc scientific name as lowercase-hyphenated
    bn_to_xc = {}
    for i, row in species_table.iterrows():
        bn_code = row.bn_code
        xc_sci_name = row.bn_mapping_in_xc_dataset
        # if both columns have a value, make a key-value pair in dictionary
        if bn_code is not np.nan and xc_sci_name is not np.nan:
            bn_to_xc[bn_code] = xc_sci_name
    xc_species_list = list(np.sort(list(bn_to_xc.values())))

    return xc_species_list


name_table_sci_idx = species_table.set_index("scientific", drop=True)
name_table_xc_com_idx = species_table.set_index("xc_common", drop=True)
name_table_bn_com_idx = species_table.set_index("bn_common", drop=True)


def sci_to_bn_common(scientific):
    """convert scientific name as lowercase-hyphenated to birdnet common name as lowercasenospaces"""
    return name_table_sci_idx.at[scientific, "bn_common"]


def sci_to_xc_common(scientific):
    """convert scientific name as lowercase-hyphenated to xeno-canto common name as lowercasenospaces"""
    return name_table_sci_idx.at[scientific, "xc_common"]


def xc_common_to_sci(common):
    """convert xeno-canto common name (ignoring dashes, spaces, case) to scientific name as lowercase-hyphenated"""
    common = common.lower().replace(" ", "").replace("-", "")
    return name_table_xc_com_idx.at[common, "scientific"]


def bn_common_to_sci(common):
    """convert bird net common name (ignoring dashes, spaces, case) to scientific name as lowercase-hyphenated"""
    common = common.lower().replace(" ", "").replace("-", "")
    return name_table_bn_com_idx.at[common, "scientific"]


def common_to_sci(common):
    """convert bird net common name (ignoring dashes, spaces, case) to scientific name as lowercase-hyphenated"""
    return bn_common_to_sci(common)
