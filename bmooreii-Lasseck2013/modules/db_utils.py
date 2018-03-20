import pymongo
import pickle
import pandas as pd
import numpy as np


def write_spectrogram(label, df, spec, normal, config):
    '''Write spectrogram to MongoDB

    Open connection to MongoDB and write the bounding box DataFrame,
    spectrogram (numpy 2D matrix), and normalization factor. The
    DataFrame and spectrogram are pickled to reduce size

    Args:
        label: The label for the MongoDB entry (the filename)
        df: The bounding box DataFrame
        spec: The numpy 2D matrix containing the spectrogram
        normal: The np.max() of the original spectrogram
        config: The openbird configuration

    Returns:
        Nothing.
    '''

    with pymongo.MongoClient(config['db_uri']) as client:
        db = client[config['db_name']]
        coll = db[config['db_collection_name']]

        # Pickle Em Up
        df_bytes = pickle.dumps(df)
        spec_bytes = pickle.dumps(spec)

        # Insert
        coll.insert_one({'label': label, 'df': df_bytes,
            'spectrogram': spec_bytes, 'normalization_factor': float(normal)})

def read_spectrogram(label, config):
    '''Read spectrogram from MongoDB

    Open connection to MongoDB and read the bounding box DataFrame,
    spectrogram (numpy 2D matrix), and normalization factor. The
    DataFrame and spectrogram are pickled to reduce size

    Args:
        label: The label for the MongoDB entry (the filename)
        config: The openbird configuration, need the uri and names

    Returns:
        Tuple containing dataframe, spectrogram, and normalization factor
    '''

    with pymongo.MongoClient(config['db_uri']) as client:
        db = client[config['db_name']]
        coll = db[config['db_collection_name']]

        # Extract DF and Spectrogram
        item = coll.find_one({'label': label})
        df_bytes = item['df']
        spec_bytes = item['spectrogram']
        normal = item['normalization_factor']
        
        # Recreate Data
        df = pd.DataFrame(pickle.loads(df_bytes))
        spec = np.array(pickle.loads(spec_bytes))

    return df, spec, normal
