import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import pymongo
from modules.spect_gen import spect_gen
from modules.image_utils import apply_gaussian_filter
from modules.db_utils import read_spectrogram


def show_or_write(image):
    '''Show or write

    Do we want to show a plot, or write an image file

    Args:
        image: If '' show plot, else write plot

    Returns:
        Nothing
    '''

    if len(image) == 0:
        plt.show()
    else:
        plt.savefig(image)


def extract_segments(spec, df):
    '''Extract the segments from a spectrogram

    Given a spectrogram and a bounding box DataFrame, extract the
    segments

    Args:
        spec: The spectrogram
        df: The bounding box DataFrame

    Returns:
        A list containing the segments
    '''
    segments = [None] * len(df.index)
    for idx in df.index:
        segments[idx] = spec[df.loc[idx]['y_min']: df.loc[idx]['y_max'],
            df.loc[idx]['x_min']: df.loc[idx]['x_max']]
    return segments


def gen_spec_with_segs(spec, df, image):
    '''Plot spectrogram w/ bounding boxes

    Plots the spectrogram with bounding boxes labeling segments

    Args:
        spec: The spectrogram
        df: The bounding box DataFrame
        image: If '' show plot, else write plot

    Returns:
        Nothing
    '''

    # Extract the segments
    segments = extract_segments(spec, df)

    # Generate Color Map
    cmap = plt.cm.get_cmap('jet', df.shape[0])
    rgb_vals = cmap(np.arange(df.shape[0]))[:, :-1]

    # Plot, flip the y-axis
    fig, ax = plt.subplots(1, figsize=(15, 5))
    ax.imshow(spec, cmap=plt.get_cmap('gray_r'))
    ax.set_ylim(ax.get_ylim()[::-1])

    for idx, row in df.iterrows():
        rect = patches.Rectangle((row['x_min'], row['y_min']),
                (row['x_max'] - row['x_min'] + 1),
                (row['y_max'] - row['y_min'] + 1),
                linewidth=1, edgecolor=rgb_vals[idx], facecolor='none')
        ax.add_patch(rect)
        ax.text(row['x_min'], row['y_min'] - 2, idx, color=rgb_vals[idx],
                fontsize=12)

    # Write or show image
    show_or_write(image)


def gen_segs(spec, df, image):
    '''Plot the segments only

    Plots the segments of the spectrogram

    Args:
        spec: The spectrogram
        df: The bounding box DataFrame
        image: If '' show plot, else write plot

    Returns:
        Nothing
    '''

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
        ax.set_ylim(ax.get_ylim()[::-1])

    # Show or write
    show_or_write(image)


def view(label, image, seg_only, config):
    '''View a spectrogram
    
    This is a super function which provides all of the functionality to
    view a spectrogram, either from the database itself or recreated

    Args:
        label: The label for the file
        image: If not '', write image to a file named this
        seg_only: View segments only
        config: The parsed ini file for this particular run

    Returns:
        Nothing. It writes to MongoDB collection defined in the ini file

    Raises:
        FileNotFoundError: If the wavfile doesn't exist, it can't be processed
    '''

    # Get the data, from the database or recreate
    df, spectrogram, normalization_factor = read_spectrogram(label, config)

    # Apply Gaussian Filter
    spectrogram = apply_gaussian_filter(spectrogram,
            config.getfloat('gaussian_filter_sigma'))

    # Either vizualize spectrogram w/ bounding boxes
    # -> or, segments
    if seg_only:
        gen_segs(spectrogram, df, image)
    else:
        gen_spec_with_segs(spectrogram, df, image)
