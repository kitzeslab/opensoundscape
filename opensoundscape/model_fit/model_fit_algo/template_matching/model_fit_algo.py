from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from opensoundscape.utils.db_utils import init_client
from opensoundscape.utils.db_utils import close_client
from opensoundscape.utils.db_utils import cursor_item_to_data
from opensoundscape.utils.db_utils import cursor_item_to_stats
from opensoundscape.utils.db_utils import read_spectrogram
from opensoundscape.utils.db_utils import return_cursor
from opensoundscape.utils.db_utils import write_file_stats
from opensoundscape.utils.db_utils import write_model
from opensoundscape.utils.db_utils import get_model_fit_skip
from opensoundscape.utils.db_utils import set_model_fit_skip
from opensoundscape.spect_gen.spect_gen import spect_gen
from opensoundscape.view.view import extract_segments
from opensoundscape.utils.image_utils import apply_gaussian_filter
from opensoundscape.utils.image_utils import generate_raw_spectrogram
from opensoundscape.utils.utils import return_cpu_count
from opensoundscape.utils.utils import get_percent_from_section
from opensoundscape.utils.utils import get_template_matching_algorithm


def min_max_vals_locs(ccorrs):
    """Given a 2D ccorrs matrix, find maxima

    Given a 2D ccorrs matrix, find the maximum cross correlation,
    bottom left location, and bottom right location. This function
    mimics cv2.minMaxLoc exporting the min and max values and location
    of those matches (i.e. the top left of where the template matched)

    Args:
        ccorrs: 2D numpy matrix of cross correlations

    Output:
        (min_ccorr, max_ccorr, (min_loc_x, min_loc_y), (max_loc_x, max_loc_y))
    """

    min_val = np.min(ccorrs)
    max_val = np.max(ccorrs)
    min_loc = np.where(ccorrs == min_val)
    max_loc = np.where(ccorrs == max_val)
    return (
        min_val,
        max_val,
        (min_loc[1][0], min_loc[0][0]),
        (max_loc[1][0], max_loc[0][0]),
    )


def should_match_templates(template_row, image_rows, freq_buffer):
    """Should OpSo even waste time matching templates?

    If there are no boxes in the source image to match against return False,
    else return True (succeed fast)

    Args:
        template_row:   A Dataframe row with keys "y_min" and "y_max"
        image_rows:     The detected boxes in an image which we might slide the template
                        against.
        freq_buffer:    The frequency buffer in the y dimension

    Returns:
        Boolean: representing whether we should match templates (True) or not (False)
    """

    match_y_min = template_row["y_min"] - freq_buffer
    match_y_max = template_row["y_max"] + freq_buffer

    for row in image_rows.iterrows():
        if (row["y_min"] >= match_y_min and row["y_min"] < match_y_max) or (
            row["y_max"] <= match_y_max and row["y_max"] > match_y_min
        ):
            return True

    return False


def crossCorrMatchTemplate(spectrogram, template):
    """Use ZNCC template matching algorithm

    Given a template and a spectrogram, slide the template along
    the spectrogram using the ZNCC method and return the cross correlations

    Args:
        spectrogram: The target spectrogram to match against
        template: The template to slide over the spectrogram

    Returns:
        cross_corrs: A 2d matrix of cross correlations
    """

    # Dimensions of the cross_corrs matrix
    o_max = spectrogram.shape[0] - template.shape[0] + 1
    i_max = spectrogram.shape[1] - template.shape[1] + 1

    output = np.zeros((o_max, i_max), dtype="float32")

    T = (template - np.mean(template)) / np.std(template)

    for o_idx in range(o_max):
        for i_idx in range(i_max):
            o_up_bound = o_idx + template.shape[0]
            i_up_bound = i_idx + template.shape[1]

            image_slice = spectrogram[o_idx:o_up_bound, i_idx:i_up_bound]
            I = (image_slice - np.mean(image_slice)) / np.std(image_slice)

            output[o_idx, i_idx] = np.sum(I * T) / I.size

    return np.nan_to_num(output)


def matchTemplate(spectrogram, template, config):
    """Use template matching to produce cross correlations

    Given a spectrogram and a template, use an algorithm to generate
    the maximum cross correlation and locations.

    Args:
        spectrogram: The target spectrogram to match against
        template: The template to slide over the spectrogram
        config: The opensoundscape config

    Returns:
        max_ccorr: The maximum cross correlation value
        max_loc_bot_left: The bottom left location of template for max_ccor
        max_loc_top_right: The top right location of template for max_ccor
    """
    method = config["model_fit"]["template_match_method"]
    if method == "opencv":
        from cv2 import matchTemplate as opencvMatchTemplate

        output_stats = opencvMatchTemplate(
            spectrogram, template, get_template_matching_algorithm(config)
        )
    elif method == "cross_corr":
        output_stats = crossCorrMatchTemplate(spectrogram, template)

    _, max_ccorr, _, (max_loc_bot_left, max_loc_top_right) = min_max_vals_locs(
        output_stats
    )
    return max_ccorr, max_loc_bot_left, max_loc_top_right


def binary_classify(correct, predicted):
    """Return type of classification

    Given the correct and predicted classification, return whether the
    prediction is a true/false positive/negative.

    Args:
        correct: the correct binary label
        predicted: the predicted binary label

    Returns:
        string

    Raises:
        Nothing.
    """

    if correct == 1 and predicted == 1:
        return "true positive"
    elif correct == 0 and predicted == 0:
        return "true negative"
    elif correct == 1 and predicted == 0:
        return "false negative"
    else:
        return "false positive"


def get_file_stats(label, config):
    """Generate the first order statistics

    Given a single label, generate the statistics for the corresponding file

    Args:
        label: A file label from the training set
        config: The parsed ini configuration

    Returns:
        The bounding box DF, spectrogram, and normalization factors for the input
        label

    Raises:
        Nothing.
    """

    # Generate the df, spectrogram, and normalization factors
    # -> Read from MongoDB or preprocess
    if config["general"].getboolean("db_rw"):
        df, spec, spec_mean, spec_std = read_spectrogram(label, config)
    else:
        df, spec, spec_mean, spec_std = spect_gen(config)

    # Generate the Raw Spectrogram
    raw_spec = generate_raw_spectrogram(spec, spec_mean, spec_std)

    # Raw Spectrogram Stats
    raw_spec_stats = stats.describe(raw_spec, axis=None)

    # Frequency Band Stats
    freq_bands = np.array_split(
        raw_spec, config["model_fit"].getint("num_frequency_bands")
    )
    freq_bands_stats = [None] * len(freq_bands)
    for idx, band in enumerate(freq_bands):
        freq_bands_stats[idx] = stats.describe(band, axis=None)

    # Segment Statistics
    df["width"] = df["x_max"] - df["x_min"]
    df["height"] = df["y_max"] - df["y_min"]
    df_stats = df[["width", "height", "y_min"]].describe()

    # Generate the file_row
    # Raw Spectrogram Stats First
    row = np.array(
        [
            raw_spec_stats.minmax[0],
            raw_spec_stats.minmax[1],
            raw_spec_stats.mean,
            raw_spec_stats.variance,
        ]
    )

    # Followed by the band statistics
    row = np.append(
        row, [[s.minmax[0], s.minmax[1], s.mean, s.variance] for s in freq_bands_stats]
    )

    # Finally the segment statistics
    # -> If the len(df_stats) == 2, it contains no segments append zeros
    if len(df_stats) == 2:
        row = np.append(row, np.zeros((3, 4)))
    else:
        row = np.append(
            row,
            (
                df_stats.loc["min"].values,
                df_stats.loc["max"].values,
                df_stats.loc["mean"].values,
                df_stats.loc["std"].values,
            ),
        )

    # The row is now a complicated object, need to flatten it
    row = np.ravel(row)

    return df, spec, spec_mean, spec_std, row


def get_file_file_stats(
    df_one, spec_one, spec_mean_one, spec_std_one, labels_df, config
):
    """Generate the second order statistics

    Given a df, spec, and normalization factors for label_one, generate the file-file statistics
    for all files (or downselect w/ template_pool.csv file)

    Args:
        monotonic_idx_one: The monotonic index for df_one
        df_one: The bounding box dataframe for label_one
        spec_one: The spectrum for label_one
        spec_mean_one: The raw spectrogram mean
        spec_std_one: The raw spectrogram standard deviation
        labels_df: All other labels
        config: The parsed ini configuration

    Returns:
        match_stats_dict: A dictionary which contains template matching statistics
         of all segments in labels_df slid over spec_one. Keys are the labels

    Raises:
        Nothing.
    """

    # Get the MongoDB Cursor, indices is a Pandas Index object -> list
    # -> If template_pool defined:
    # -> 1. Generate a pools dataframe and convert string to [int]
    # -> 2. Read items from template_pool_db if necessary
    if config["general"].getboolean("db_rw"):
        if config["model_fit"]["template_pool"]:
            pools_df = pd.read_csv(config["model_fit"]["template_pool"], index_col=0)
            pools_df.templates = pools_df.templates.apply(lambda x: json.loads(x))

            if config["model_fit"]["template_pool_db"]:
                items = return_cursor(
                    pools_df.index.values.tolist(),
                    "spectrograms",
                    config,
                    config["model_fit"]["template_pool_db"],
                )
            else:
                items = return_cursor(
                    pools_df.index.values.tolist(), "spectrograms", config
                )
        else:
            items = return_cursor(
                labels_df.index.values.tolist(), "spectrograms", config
            )
    else:
        items = {"label": x for x in labels_df.index.values.tolist()}

    match_stats_dict = {}

    spec_one = apply_gaussian_filter(
        spec_one, config["model_fit"]["gaussian_filter_sigma"]
    )

    # Iterate through the cursor
    for item in items:
        # Need to get the index for match_stats
        idx_two = item["label"]
        # monotonic_idx_two, = np.where(get_segments_from == idx_two)
        # monotonic_idx_two = monotonic_idx_two[0]

        if config["general"].getboolean("db_rw"):
            df_two, spec_two, spec_mean_two, spec_std_two = cursor_item_to_data(
                item, config
            )
        else:
            df_two, spec_two, spec_mean_two, spec_std_two = spect_gen(config)

        spec_two = apply_gaussian_filter(
            spec_two, config["model_fit"]["gaussian_filter_sigma"]
        )

        # Extract segments
        # -> If using template_pool, downselect the dataframe before extracting segments
        if config["model_fit"]["template_pool"]:
            df_two = df_two.iloc[pools_df.loc[idx_two].values[0]]
        df_two["segments"] = extract_segments(spec_two, df_two)

        # Generate the np.array to append
        match_stats_dict[idx_two] = np.zeros((df_two.shape[0], 3))

        # Slide segments over all other spectrograms
        frequency_buffer = config["model_fit"].getint("template_match_frequency_buffer")
        for idx, (_, row) in enumerate(df_two.iterrows()):

            # Determine minimum y target
            y_min_target = 0
            if row["y_min"] > frequency_buffer:
                y_min_target = row["y_min"] - frequency_buffer

            # Determine maximum y target
            y_max_target = spec_two.shape[0]
            if row["y_max"] < spec_two.shape[0] - frequency_buffer:
                y_max_target = row["y_max"] + frequency_buffer

            # If the template is small enough, do the following:
            # -> Match the template against the stripe of spec_one with the 5th
            # -> algorithm of matchTemplate, then grab the max correllation
            # -> max location x value, and max location y value
            if (
                y_max_target - y_min_target <= spec_one.shape[0]
                and row["x_max"] - row["x_min"] <= spec_one.shape[1]
            ):
                # If `only_match_if_detected_boxes` == False, just match as normal
                #                                      True, rely on should_match_templates
                match_anyway = not config["model_fit"].getboolean(
                    "only_match_if_detected_boxes"
                )
                if match_anyway or should_match_templates(
                    row, df_one, frequency_buffer
                ):
                    max_val, max_loc_bot_left, max_loc_top_right = matchTemplate(
                        spec_one[y_min_target:y_max_target, :], row["segments"], config
                    )
                    match_stats_dict[idx_two][idx][0] = max_val
                    match_stats_dict[idx_two][idx][1] = max_loc_bot_left
                    match_stats_dict[idx_two][idx][2] = max_loc_top_right + y_min_target
    return match_stats_dict


def chunk_run_stats(chunk, labels_df, config):
    """For each chunk call run_stats

    Run within a parallel executor to generate file and file-file Statistics
    for a given label.

    Args:
        chunk: A chunk of file labels
        labels_df: Passed through to `file_file_stats` function
        config: The parsed ini file for this run

    Returns:
        Nothing, writes to MongoDB

    Raises:
        Nothing.
    """

    init_client(config)

    [run_stats(label, labels_df, config) for label in chunk]

    close_client()

    return


def run_stats(idx_one, labels_df, config):
    """Wrapper for parallel stats execution

    Run within a parallel executor to generate file and file-file Statistics
    for a given label.

    Args:
        idx_one: The label for the file
        labels_df: Passed through to `file_file_stats` function
        config: The parsed ini file for this run

    Returns:
        Nothing, writes to MongoDB

    Raises:
        Nothing.
    """
    df_one, spec_one, spec_mean_one, spec_std_one, row_f = get_file_stats(
        idx_one, config
    )
    match_stats = get_file_file_stats(
        df_one, spec_one, spec_mean_one, spec_std_one, labels_df, config
    )
    write_file_stats(idx_one, row_f, match_stats, config)


def build_X_y(labels_df, config):
    """Build X and y from labels_df

    Build X and y to fit a DecisionTree Classifier

    Args:
        labels_df: labels dataframe for this particular bird

    Returns:
        X: dataframe containing data to fit on
        y: series containing the labels

    Raises:
        Nothing.
    """

    if config["model_fit"]["template_pool"]:
        pools_df = pd.read_csv(config["model_fit"]["template_pool"], index_col=0)
        pools_df.templates = pools_df.templates.apply(lambda x: json.loads(x))
        get_file_file_stats_for = [x for x in pools_df.index]
    else:
        get_file_file_stats_for = [x for x in labels_df.index if labels_df[x] == 1]

    items = return_cursor(labels_df.index.values.tolist(), "statistics", config)

    # What we need is numpy arrays for file_stats and file_file_stats
    # Issue: The dimensionality for file_file_stats[mono_idx] is
    # -> (num_labeled_files, num_templates_in_labeled_file, 3)
    # -> num_templates_in_labeled_file varies, therefore we need a
    # -> np.vstack to collapse them to (num_labeled_files * num_templates_in_labeled_file, 3)
    # -> to create our final file_file_stats with dimensions:
    # -> (num_files, num_labeled_files * num_templates_in_labeled_file, 3)
    file_stats = [None] * labels_df.shape[0]
    file_file_stats = [None] * labels_df.shape[0]
    for item in items:
        mono_idx = labels_df.index.get_loc(item["label"])
        file_stats[mono_idx], file_file_stats[mono_idx] = cursor_item_to_stats(item)
        file_file_stats[mono_idx] = np.vstack(
            [file_file_stats[mono_idx][x] for x in get_file_file_stats_for]
        )

    # Shape: (num_files, 80), 80 is number of file statistics
    # -> sometimes garbage data in file_stats (i.e. need nan_to_num)
    file_stats = np.nan_to_num(np.array(file_stats))

    # Input Shape: [num_files, np.array(num_templates, num_features)]
    # Output Shape: np.array(num_files, num_templates, num_features)
    file_file_stats = np.array(file_file_stats)

    # Short circuit return for only cross correlations
    if config["model_fit"].getboolean("cross_correlations_only"):
        return (
            pd.DataFrame(
                file_file_stats[:, :, 0].reshape(file_file_stats.shape[0], -1)
            ),
            pd.Series(labels_df.values),
        )

    return (
        pd.DataFrame(
            np.hstack(
                (file_stats, file_file_stats.reshape(file_file_stats.shape[0], -1))
            )
        ),
        pd.Series(labels_df.values),
    )


def fit_model(X, y, labels_df, config):
    """Fit model on X, y

    Given X, y perform train/test split, scaling, and model fitting

    Args:
        X: dataframe containing model data
        y: labels series
        labels_df: labels for this run
        config: the config for this run

    Returns:
        model: The sklearn model

    Raises:
        Nothing.
    """
    test_size = get_percent_from_section(config, "model_fit", "stratification_percent")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = RandomForestClassifier()

    params = {
        "n_estimators": [config["model_fit"].getint("n_estimators")],
        "max_features": [config["model_fit"].getint("max_features")],
        "min_samples_split": [config["model_fit"].getint("min_samples_split")],
    }

    grid_search = GridSearchCV(classifier, params, cv=y_train.sum(), n_jobs=1)
    grid_search.fit(X_train, y_train)

    y_train_pred = grid_search.best_estimator_.predict(X_train)
    y_test_pred = grid_search.best_estimator_.predict(X_test)

    train_filenames = labels_df.iloc[y_train.index.values].index.values
    test_filenames = labels_df.iloc[y_test.index.values].index.values

    train_classify = [
        f"{binary_classify(actual, pred)}"
        for name, actual, pred in zip(train_filenames, y_train, y_train_pred)
    ]

    test_classify = [
        f"{binary_classify(actual, pred)}"
        for name, actual, pred in zip(test_filenames, y_test, y_test_pred)
    ]

    train_false_pos = " ".join(
        [
            name
            for name, classifier in zip(train_filenames, train_classify)
            if classifier == "false positive"
        ]
    )

    train_false_neg = " ".join(
        [
            name
            for name, classifier in zip(train_filenames, train_classify)
            if classifier == "false negative"
        ]
    )

    test_false_pos = " ".join(
        [
            name
            for name, classifier in zip(test_filenames, test_classify)
            if classifier == "false positive"
        ]
    )

    test_false_neg = " ".join(
        [
            name
            for name, classifier in zip(test_filenames, test_classify)
            if classifier == "false negative"
        ]
    )

    results = [
        f"{roc_auc_score(y_train, y_train_pred)}",
        f"{roc_auc_score(y_test, y_test_pred)}",
        f"{precision_score(y_train, y_train_pred)}",
        f"{precision_score(y_test, y_test_pred)}",
        f"{recall_score(y_train, y_train_pred)}",
        f"{recall_score(y_test, y_test_pred)}",
        f"{f1_score(y_train, y_train_pred)}",
        f"{f1_score(y_test, y_test_pred)}",
        f'"{confusion_matrix(y_train, y_train_pred).tolist()}"',
        f'"{confusion_matrix(y_test, y_test_pred).tolist()}"',
        f"{train_false_neg}",
        f"{train_false_pos}",
        f"{test_false_neg}",
        f"{test_false_pos}",
    ]

    return grid_search.best_estimator_, scaler, results


def chunk_build_model(chunk, labels_df, config):

    """Build the model
    Given a chunk, run build_model on each label

    Args:
        chunk: Some columns to process in parallel
        labels_df: The labels_df to build model with
        config: The parsed ini file for this run

    Returns:
        Something or possibly writes to MongoDB

    Raises:
        Nothing.
    """

    init_client(config)

    results = [build_model(col, labels_df, config) for col in chunk]

    close_client()

    return results


def build_model(column, labels_df, config):
    """Build the model

    We were directed here from model_fit to fit a template matching model.

    Args:
        column: The column to build a model with
        labels_df: The labels for the column
        config: The parsed ini file for this run

    Returns:
        Something or possibly writes to MongoDB

    Raises:
        Nothing.
    """
    X, y = build_X_y(labels_df[column], config)
    model, scaler, results = fit_model(X, y, labels_df, config)
    write_model(column, model, scaler, config)
    return column, results


def model_fit_algo(config):
    """Fit the model

    We were directed here from model_fit to fit the model.

    Args:
        config: The parsed ini file for this run

    Returns:
        Something or possibly writes to MongoDB

    Raises:
        Nothing.
    """

    # First, we need labels and files
    labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['train_file']}",
        index_col=0,
    )
    labels_df = labels_df.fillna(0).astype(int)

    if config["model_fit"]["species_list"] != "":
        labels_df = labels_df.loc[:, config["model_fit"]["species_list"].split(",")]

    # Get the processor counts
    nprocs = return_cpu_count(config)

    # Define the parallel executor
    executor = ProcessPoolExecutor(nprocs)

    # Run the statistics, if not already complete
    chunks = np.array_split(labels_df.index, nprocs)
    if not get_model_fit_skip(config) or config["docopt"].getboolean(
        "rerun_statistics"
    ):
        fs = [
            executor.submit(chunk_run_stats, chunk, labels_df, config)
            for chunk in chunks
        ]
        # This shouldn't be necessary, but doesn't work otherwise...
        for future in as_completed(fs):
            _ = future.result()

    set_model_fit_skip(config)

    # Column Labels
    metric_labels = [
        "ROC AUC Train",
        "ROC AUC Test",
        "Precision Train",
        "Precision Test",
        "Recall Train",
        "Recall Test",
        "F1 Score Train",
        "F1 Score Test",
        "Confusion Matrix Train",
        "Confusion Matrix Test",
        "Train False Negative",
        "Train False Positive",
        "Test False Negative",
        "Test False Positive",
    ]

    # Build the models
    chunks = np.array_split(labels_df.columns, nprocs)
    fs = [
        executor.submit(chunk_build_model, chunk, labels_df, config) for chunk in chunks
    ]
    # Start a csv file if defined
    # -> "x" fails if the file exists!
    if config["docopt"]["csv_file"]:
        with open(config["docopt"]["csv_file"], "x") as csv:
            csv.write(f"Label,{','.join(metric_labels)}\n")
    for future in as_completed(fs):
        result = future.result()
        if result:
            for column, metrics in result:
                # Append to csv, if defined
                if config["docopt"]["csv_file"]:
                    with open(config["docopt"]["csv_file"], "a") as csv:
                        csv.write(f"{column},{','.join(metrics)}\n")

                # Print metrics to command line
                print(f"Label: {column}")
                print("=" * 30)
                for label, metric in zip(metric_labels, metrics):
                    print(f"{label}: {metric}")
