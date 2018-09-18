import pymongo
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from datetime import datetime
from modules.utils import yes_no

# This is a thread local global variable
client = None

class OpensoundscapeAttemptOverrideINISection(Exception):
    '''Override INI Section Exception
    '''
    pass


def init_client(config):
    '''Initialize a MongoDB Client

    Initialize the MongoDB Client

    Args:
        config: The opensoundscape configuration

    Returns:
        Nothing.

    Raises:
        Nothing
    '''

    global client
    client = pymongo.MongoClient(config['general']['db_uri'])


def close_client():
    '''Close MongoDB Client Connection

    Close the MongoDB Client Connection

    Args:
        config: The opensoundscape configuration

    Returns:
        Nothing.

    Raises:
        Nothing
    '''

    global client
    client.close()


def write_ini_section(config, section):
    '''Write the ini to database

    Open connection to MongoDB and write the INI file section to the database.
    The db section will operate like a lock file. Each section is reliant on the
    'general' section so we check that for all other sections. If you try to
    change the config, we will warn the user it is happening.

    Args:
        config: The opensoundscape configuration
        section: Which database section to work on

    Returns:
        Nothing.

    Raises:
        OpensoundscapeAttemptOverrideINISection: When INI section exists and user
            doesn't override
    '''

    global client
    init_client(config)

    ini_dict = dict(config[section].items())
    gen_dict = dict(config['general'].items())

    db = client[config['general']['db_name']]
    coll = db['ini']

    if section != 'general':
        item = coll.find_one({'section': 'general'})
        if item == None:
            raise OpensoundscapeAttemptOverrideINISection("Please run `opensoundscape.py init`")
        else:
            item.pop('section')
            item.pop('_id')
            if gen_dict != item:
                raise OpensoundscapeAttemptOverrideINISection("Please run `opensoundscape.py init`")

    item = coll.find_one({'section': section})
    if item == None:
        ini_dict['section'] = section
        coll.insert_one(ini_dict)
        if section == 'model_fit':
            db['model_fit_statistics_skip'].insert_one({'skip': False})
    else:
        item.pop('section')
        item.pop('_id')
        if ini_dict != item:
            print("WARNING: Detected a change to your configuration!")
            answer = yes_no("Do you want to override the current config?")
            if not answer:
                raise OpensoundscapeAttemptOverrideINISection("Check your INI file!")
            else:
                coll.update_one({'section': section}, {'$set': ini_dict},
                    upsert=True)
                if section == 'model_fit':
                    db['model_fit_statistics_skip'].update_one({},
                        {'$set': {'skip': False}}, upsert=True)

    # Make sure to close the client!
    close_client()


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
        config: The opensoundscape configuration

    Returns:
        Nothing.
    '''
    global client

    df_bytes = pickle.dumps(df)

    if config['general'].getboolean('db_sparse'):
        spec[spec <
                (config['general'].getfloat('db_sparse_thresh_percent') / 100.)] = 0
        spec_bytes = pickle.dumps(csr_matrix(spec))
    else:
        spec_bytes = pickle.dumps(spec)

    db = client[config['general']['db_name']]
    coll = db['spectrograms']
    coll.update_one({'label': label},
        {'$set': {'df': df_bytes, 'spectrogram': spec_bytes,
            'normalization_factor': normal, 'preprocess_date': datetime.now()}},
        upsert=True)


def read_spectrogram(label, config):
    '''Read spectrogram from MongoDB

    Open connection to MongoDB and read the bounding box DataFrame, spectrogram
    (compressed sparse row 2D matrix), and normalization factor. The DataFrame
    and spectrogram are pickled to reduce size

    Args:
        label: The label for the MongoDB entry (the filename)
        config: The opensoundscape configuration, need the uri and names

    Returns:
        Tuple containing dataframe, spectrogram (dense representation), and
        normalization factor
    '''

    global client

    db = client[config['general']['db_name']]
    coll = db['spectrograms']

    # Extract DF and Spectrogram
    item = coll.find_one({'label': label})
    df, spec, normal = cursor_item_to_data(item, config)

    return df, spec, normal


def return_cursor(indices, coll, config, db_name=''):
    '''Generate a cursor of all other spectrograms

    Open connection to MongoDB, generate a cursor with the list of indices,
    return it

    Args:
        indices: A list of the labels to return
        coll: The collection to draw items from
        config: The opensoundscape configuration, need the uri and names
        db_name: Draw items from a different database

    Returns:
        MongoDB Cursor
    '''

    global client

    if db_name:
        db = client[db_name]
    else:
        db = client[config['general']['db_name']]
    coll = db[coll]

    items = coll.find({'label': {'$in': indices}})
    return items


def cursor_item_to_data(item, config):
    '''Given an item, return necessary spectrogram data

    Utility function to convert an item in the database to bounding
    box dataframe, spectogram, and normalization factor.

    Args:
        item: A database item
        config: The opensoundscape configuration

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
    if config['general'].getboolean('db_sparse'):
        spec = spec.todense()
    return df, spec, normal


def cursor_item_to_stats(item):
    '''Given an item, return file and file_file statistics

    Utility function to convert an item in the database to file and file_file
    statistics

    Args:
        item: A database item

    Returns:
        file_stats: single file statistics
        file_file_stats: file-file statistics
    '''
    file_stats_bytes = item['file_stats']
    file_file_stats_bytes = item['file_file_stats']

    return pickle.loads(file_stats_bytes), pickle.loads(file_file_stats_bytes)


def write_file_stats(label, file_stats, file_file_stats, config):
    '''Write file statistics to MongoDB

    Open connection to MongoDB and write the pickled file statistics

    Args:
        label: The label for the MongoDB entry
        file_stats: The file stats as a numpy array
        file_file_stats: An array containing file file statistics for training files
        config: The opensoundscape configuration

    Returns:
        Nothing.
    '''

    global client

    # Pickle it up
    file_stats_bytes = pickle.dumps(file_stats)
    file_file_stats_bytes = pickle.dumps(file_file_stats)

    # Update or insert item into collection
    db = client[config['general']['db_name']]
    coll = db['statistics']
    coll.update_one({'label': label},
        {'$set': {'file_stats': file_stats_bytes,
            'file_file_stats': file_file_stats_bytes}},
        upsert=True)


def write_model(label, model, scaler, config):
    '''Write model to MongoDB

    Open connection to MongoDB and write the model

    Args:
        label: The label for the MongoDB entry
        model: The sklearn model
        config: The opensoundscape configuration

    Returns:
        Nothing.
    '''

    global client

    model_bytes = pickle.dumps(model)
    scaler_bytes = pickle.dumps(scaler)

    db = client[config['general']['db_name']]
    coll = db['models']
    coll.update_one({'label': label}, {'$set': {'model': model_bytes,
        'scaler': scaler_bytes}}, upsert=True)


def recall_model(label, config):
    '''Recall model from MongoDB

    Open connection to MongoDB and recall the model

    Args:
        label: The label for the MongoDB entry
        config: The opensoundscape configuration

    Returns:
        Nothing.
    '''

    global client

    db = client[config['general']['db_name']]
    coll = db['models']
    item = coll.find_one({'label': label})
    return pickle.loads(item['model']), pickle.loads(item['scaler'])


def get_model_fit_skip(config):
    '''Get model_fit_statistics_skip

    Open connection to MongoDB and get model_fit_statistics_skip

    Args:
        config: The opensoundscape configuration

    Returns:
        Boolean
    '''

    global client

    db = client[config['general']['db_name']]
    coll = db['model_fit_statistics_skip']
    return coll.find_one({})['skip']


def set_model_fit_skip(config):
    '''Set model_fit_statistics_skip to True

    Open connection to MongoDB and set model_fit_statistics_skip to True so
    we can generate models more quickly after statsistics have been run

    Args:
        config: The opensoundscape configuration

    Returns:
        Nothing
    '''

    global client

    db = client[config['general']['db_name']]
    coll = db['model_fit_statistics_skip']
    coll.update_one({}, {'$set': {'skip': True}}, upsert=True)
