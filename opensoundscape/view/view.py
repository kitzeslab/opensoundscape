import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import pymongo
from opensoundscape.spect_gen.spect_gen import spect_gen
from opensoundscape.utils.image_utils import generate_raw_blurred_spectrogram
from opensoundscape.utils.db_utils import init_client
from opensoundscape.utils.db_utils import close_client
from opensoundscape.utils.db_utils import read_spectrogram


def show_or_write(image):
    """Show or write

    Do we want to show a plot, or write an image file

    Args:
        image: If '' show plot, else write plot

    Returns:
        Nothing
    """

    if image:
        plt.savefig(image)
    else:
        plt.show()


def extract_segments(spec, df):
    """Extract the segments from a spectrogram

    Given a spectrogram and a bounding box DataFrame, extract the
    segments

    Args:
        spec: The spectrogram
        df: The bounding box DataFrame

    Returns:
        A list containing the segments
    """
    segments = [None] * len(df.index)
    for seg_idx, idx in enumerate(df.index):
        segments[seg_idx] = spec[
            df.loc[idx]["y_min"] : df.loc[idx]["y_max"],
            df.loc[idx]["x_min"] : df.loc[idx]["x_max"],
        ]
    return segments


def gen_spec_with_segs(label, spec, df, image):
    """Plot spectrogram w/ bounding boxes

    Plots the spectrogram with bounding boxes labeling segments

    Args:
        label: The spectrogram title
        spec: The spectrogram
        df: The bounding box DataFrame
        image: If '' show plot, else write plot

    Returns:
        Nothing
    """

    # Extract the segments
    segments = extract_segments(spec, df)

    # Generate Color Map
    if df.shape[0] != 0:
        cmap = plt.cm.get_cmap("jet", df.shape[0])
        rgb_vals = cmap(np.arange(df.shape[0]))[:, :-1]

    # Plot, flip the y-axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(spec, cmap=plt.get_cmap("gray_r"))
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(label)
    ax.set_aspect(spec.shape[1] / (2 * spec.shape[0]))

    for idx, row in df.iterrows():
        rect = patches.Rectangle(
            (row["x_min"], row["y_min"]),
            (row["x_max"] - row["x_min"] + 1),
            (row["y_max"] - row["y_min"] + 1),
            linewidth=1,
            edgecolor=rgb_vals[idx],
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(row["x_min"], row["y_min"] - 2, idx, color=rgb_vals[idx], fontsize=12)

    # Write or show image
    show_or_write(image)


def gen_segs(spec, df, image):
    """Plot the segments only

    Plots the segments of the spectrogram

    Args:
        spec: The spectrogram
        df: The bounding box DataFrame
        image: If '' show plot, else write plot

    Returns:
        Nothing
    """

    # Extract the segments
    segments = extract_segments(spec, df)

    # Determine the shape constaints
    max_rows = 5
    columns = (df.index.shape[0] // max_rows) + (df.index.shape[0] % max_rows)

    # Create the figure,
    # -> for each segment, add subplot
    fig = plt.figure()
    for idx in df.index:
        ax = fig.add_subplot(max_rows, columns, idx + 1)
        ax.imshow(segments[idx], cmap=plt.get_cmap("gray_r"))
        ax.set_title(idx)
        ax.set_ylim(ax.get_ylim()[::-1])

    # Show or write
    show_or_write(image)


def view(config):
    """View a spectrogram

    This is a super function which provides all of the functionality to
    view a spectrogram, either from the database itself or recreated

    Args:
        config: The parsed ini file for this particular run

    Returns:
        Nothing. It writes to MongoDB collection defined in the ini file

    Raises:
        FileNotFoundError: If the wavfile doesn't exist, it can't be processed
    """

    # Need some command line arguments from config
    label = config["docopt"]["label"]
    image = config["docopt"]["image"]
    seg_only = config["docopt"].getboolean("print_segments")

    # Get the data, from the database or recreate
    init_client(config)
    df, spectrogram, spectrogram_mean, spectrogram_std = read_spectrogram(label, config)
    close_client()

    spectrogram = generate_raw_blurred_spectrogram(
        spectrogram,
        spectrogram_mean,
        spectrogram_std,
        config["model_fit"]["gaussian_filter_sigma"],
    )

    # Either vizualize spectrogram w/ bounding boxes
    # -> or, segments
    if seg_only:
        gen_segs(spectrogram, df, image)
    else:
        gen_spec_with_segs(label, spectrogram, df, image)
