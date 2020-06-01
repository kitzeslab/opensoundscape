from opensoundscape.config.checks import ini_section_and_keys_exists
from opensoundscape.config.checks import config_checks
from os.path import isfile
from configparser import ConfigParser


DEFAULT_CONFIG = """
[general]
num_processors =
db_rw = True
db_name = opensoundscape
db_uri = localhost:27017
db_sparse = False
db_sparse_thresh_percent = 1.0
data_dir =
train_file =
validate_file =
predict_file =

[spect_gen]
algo = template_matching
resample_rate = 22050.0
# Options: kaiser_best, kaiser_fast, scipy
resample_type = kaiser_best
spectrogram_segment_length = 512
spectrogram_overlap = 75
low_freq_thresh = 0
high_freq_thresh = 0
decibel_threshold = -100.0
median_filter_factor = 0.25
low_values_filter_percent = 1.5
binary_closing_kernel_height = 6
binary_closing_kernel_width = 10
binary_dilation_kernel_height = 3
binary_dilation_kernel_width = 5
median_filter_kernel_height = 5
median_filter_kernel_width = 3
small_objects_kernel_size = 50
segment_pixel_buffer = 12

[model_fit]
algo = template_matching
num_frequency_bands = 16
gaussian_filter_sigma = 1.5
template_match_frequency_buffer = 5
template_pool =
template_pool_db =
n_estimators = 1
max_features = 4
min_samples_split = 3
cross_correlations_only = False
species_list =
stratification_percent = 33.3
# Options: opencv, cross_corr
template_match_method = opencv
# Options: TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
template_match_algorithm = TM_CCORR_NORMED
only_match_if_detected_boxes = False

[predict]
algo = template_matching
"""


def generate_config(arguments, store_options=True):
    """Generate the configuration

    Simply return a ConfigParser for opensoundscape. We have a default config in
    `config/` as well as a potential override file (arguments["--ini"]). Access elements via
    `config[<section>].get{float,boolean,int}('key')`.

    Args:
        arguments: The docopt arguments to store

    Returns:
        A ConfigParser instance

    Raises:
        FileNotFoundError if INI file doesn't exist
    """
    f_override = arguments["--ini"]

    if not isfile(f_override):
        raise FileNotFoundError(f"{f_override} doesn't exist!")

    config = ConfigParser(allow_no_value=True)
    config.read_string(DEFAULT_CONFIG)

    override_config = ConfigParser()
    override_config.read(f_override)

    # Check that the override config makes sense, then read it
    ini_section_and_keys_exists(config, override_config, f_override)
    config.read(f_override)

    config_checks(config)

    if store_options:
        config["docopt"] = {}
        config["docopt"]["label"] = arguments["<label>"]
        config["docopt"]["image"] = arguments["<image>"]
        config["docopt"]["print_segments"] = str(arguments["--segments"]).lower()
        config["docopt"]["rerun_statistics"] = str(
            arguments["--rerun_statistics"]
        ).lower()
        config["docopt"]["csv_file"] = arguments["<csv_file>"]

    return config
