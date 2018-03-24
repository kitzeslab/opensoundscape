def sampling(dir, config):
    '''Sample labeled data

    Given a directory of labeled data, sample it to form train/test data.

    Args:
        dir: A directory containing labeled data
        config: The parsed ini file for this run

    Returns:
        Nothing. Write metadata to MongoDB collection indicating train/test
        data.

    Raises:
        NotImplementedError: Not written yet.
    '''
    raise NotImplementedError("Sampling functionality isn't available")
