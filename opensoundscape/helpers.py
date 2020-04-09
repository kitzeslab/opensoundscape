def isNan(x):
    return not x==x

def bound(x, bounds):
    """ restrict x to a range of bounds = [min, max]"""
    return min( max(x,bounds[0]), bounds[1] )

def binarize(x, threshold):
    """ return a list of 0, 1 by thresholding vector x """
    return [1 if xi > threshold else 0 for xi in x]

def run_command(cmd):
    """ run a bash command with Popen, return response"""
    from subprocess import Popen, PIPE
    from shlex import split

    return Popen(split(cmd), stdout=PIPE, stderr=PIPE).communicate()

def rescale_features(X,rescaling_vector = None):
    """ rescale all features by dividing by the max value for each feature
    
    optionally provide the rescaling vector (1xlen(X) np.array), 
    so that you can rescale a new dataset consistently with an old one
    
    returns rescaled feature set and rescaling vector"""    
    import numpy as np
    
    if rescaling_vector is None:
        rescaling_vector = 1/np.nanmax(X,0)
    rescaledX = np.multiply(X,rescaling_vector).tolist()
    return rescaledX, rescaling_vector

def file_name(path):
    import os
    return os.path.splitext(os.path.basename(path))[0]

def hex_to_time(s):
    from datetime import datetime
    sec = int(s,16)
    timestamp = datetime.fromtimestamp(sec)
    return timestamp