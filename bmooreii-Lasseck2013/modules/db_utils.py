import pymongo
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from datetime import datetime


def write_spectrogram(label, df, spec, normal, config):
    '''Write spectrogram to MongoDB

    Open connection to MongoDB and write the bounding box DataFrame,
    spectrogram (compressed sparse row 2D matrix), and normalization
    factor.  The DataFrame and spectrogram are pickled to reduce size

    Args:
        label: The label for the MongoDB entry (the filename)
        df: The bounding box DataFrame
        spec: The numpy 2D matrix containing the spectrogram
        normal: The np.max() of the original spectrogram
        config: The openbird configuration

    Returns:
        Nothing.
    '''

    # Pickle Up the DF
    df_bytes = pickle.dumps(df)

    # Steps:
    # 1. Set the lowest 5% values to zero
    # 2. Store as compressed sparse row matrix
    # 3. Pickle and store
    if config.getboolean('db_sparse'):
        spec[spec <
                (config.getfloat('db_sparse_thresh_percent') / 100.)] = 0
        spec_bytes = pickle.dumps(csr_matrix(spec))
    else:
        spec_bytes = pickle.dumps(spec)

    # Update or insert item into collection
    with pymongo.MongoClient(config['db_uri']) as client:
        db = client[config['db_name']]
        coll = db[config['db_collection_name']]
        coll.update_one({'data_dir': config['data_dir'], 'label': label},
                {'$set': {'df': df_bytes, 'spectrogram': spec_bytes,
                    'normalization_factor': normal, 'sparse':
                    config.getboolean('db_sparse'), 'sparse_thresh_percent':
                    config.getfloat('db_sparse_thresh_percent'),
                    'spect_gen_preprocess_method':
                    config['spect_gen_preprocess_method'], 'preprocess_date':
                    datetime.now()}}, upsert=True)


def read_spectrogram(label, config):
    '''Read spectrogram from MongoDB

    Open connection to MongoDB and read the bounding box DataFrame, spectrogram
    (compressed sparse row 2D matrix), and normalization factor. The DataFrame
    and spectrogram are pickled to reduce size

    Args:
        label: The label for the MongoDB entry (the filename)
        config: The openbird configuration, need the uri and names

    Returns:
        Tuple containing dataframe, spectrogram (dense representation), and
        normalization factor
    '''

    with pymongo.MongoClient(config['db_uri']) as client:
        db = client[config['db_name']]
        coll = db[config['db_collection_name']]

        # Extract DF and Spectrogram
        item = coll.find_one({'data_dir': config['data_dir'], 'label': label})
        df, spec, normal = cursor_item_to_data(item, config)

    return df, spec, normal


def return_spectrogram_cursor(indices, config):
    '''Generate a cursor of all other spectrograms

    Open connection to MongoDB, generate a cursor with the list of indices,
    return it

    Args:
        indices: A list of the labels to return
        config: The openbird configuration, need the uri and names

    Returns:
        MongoDB Cursor
    '''

    with pymongo.MongoClient(config['db_uri']) as client:
        db = client[config['db_name']]
        coll = db[config['db_collection_name']]

        items = coll.find({'data_dir': config['data_dir'], 'label': {'$in': indices}})
        return items


def cursor_item_to_data(item, config):
    '''Given an item, return necessary spectrogram data

    Utility function to convert an item in the database to bounding
    box dataframe, spectogram, and normalization factor.

    Args:
        item: A database item
        config: The openbird configuration

    Returns:
        df: The bounding box dataframe,
        spec: The dense spectrogram
        normal: The normalization factor
    '''
    df_bytes = item['df']
    spec_bytes = item['spectrogram']
    normal = item['normalization_factor']

    # Recreate Data
    df = pd.DataFrame(pickle.loads(df_bytes))
    spec = pickle.loads(spec_bytes)
    if config.getboolean('db_sparse'):
        spec = spec.todense()
    return df, spec, normal


def write_file_stats(label, file_stats, file_file_stats, config):
    '''Write file statistics to MongoDB

    Open connection to MongoDB and write the pickled file statistics

    Args:
        label: The label for the MongoDB entry
        file_stats: The file stats as a numpy array
        file_file_stats: An array containing file file statistics for training files
        config: The openbird configuration

    Returns:
        Nothing.
    '''

    # Pickle it up
    file_stats_bytes = pickle.dumps(file_stats)
    file_file_stats_bytes = pickle.dumps(file_file_stats)

    # Update or insert item into collection
    with pymongo.MongoClient(config['db_uri']) as client:
        db = client[config['db_name']]
        coll = db[config['db_collection_name']]
        coll.update_one({'data_dir': config['data_dir'], 'label': label},
            {'$set': {'file_stats': file_stats_bytes,
            'file_file_stats': file_file_stats_bytes}})
