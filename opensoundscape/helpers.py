def isNan(x):
    return not x==x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    timestamp = datetime.utcfromtimestamp(sec)
    return timestamp

def min_max_scale(spect, feature_range=(0, 1)):
    bottom, top = feature_range
    spect_min = spect.min()
    spect_max = spect.max()
    scale_factor = (top - bottom) / (spect_max - spect_min)
    return scale_factor * (spect - spect_min) + bottom

def jitter(
            x,
            width,
            distribution = 'gaussian', #or 'uniform'
            ):
    
    '''
    Jitter (add random noise to) each value of x
    
    Inputs:
        x: scalar, array, or nd-array of numeric type
        width: multiplier for random variable (stdev for 'gaussian' or r for 'uniform')
        distribution: 'gaussian' (default) or 'uniform'
            if 'gaussian': draw jitter from gaussian with mu = 0, std = width
            if 'uniform': draw jitter from uniform on [-width, width]    
    Returns: 
        jittered_x: x + random jitter
    '''        
    if distribution == 'gaussian':
        return np.array(x) + np.random.normal(0,width,size=np.shape(x),)
    elif distribution == 'uniform':
        return np.array(x) + np.random.uniform(-1*width,width,size=np.shape(x))
    raise ValueError(f"distribution must be 'gaussian' or 'uniform'. Got {distribution}.")