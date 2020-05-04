from librosa import load 
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt 
import scipy
import os

def run_command(cmd):
    from subprocess import Popen, PIPE
    from shlex import split
    return Popen(split(cmd), stdout=PIPE, stderr=PIPE).communicate()

#move to Audio or Librosa scripts
def audio_gate(source_path,destination_path,cutoff = -38):
    """perform audio gate with ffmpeg and save new audio file
    
    audio gate refers to muting the file except when amplitude exceeds a threshold"""
    gatingCmd = (f'ffmpeg -i {source_path} "compand='
        + f'attacks=0.1:points=-115/-115|{float(cutoff) - 0.1}/-115|{cutoff}/{cutoff}|20/20" {destination_path} ')
    
    return run_command(cmd)
    
# def bandpass_filter(signal, lowcut, highcut, sample_rate, order=9):
#     """perform a butterworth bandpass filter on a discrete time signal
#     using scipy.signal's butter and lfilter
    
#     signal: discrete time signal (audio samples, list of float)
#     low_f: -3db point for highpass filter (Hz)
#     high_f: -3db point for highpass filter (Hz)
#     sample_rate: samples per second (Hz)
#     order=9: higher values -> steeper dropoff
    
#     return: filtered time signal 
#     """
#     from scipy.signal import butter, lfilter

#     nyq = 0.5 * sample_rate
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')        
    
#     return lfilter(b, a, signal)

from scipy.signal import butter, sosfiltfilt, sosfreqz

def butter_bandpass(low_f, high_f, sample_rate, order=9):
    """generate coefs for bandpass filter"""
    nyq = 0.5 * sample_rate
    low = low_f / nyq
    high = high_f / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def bandpass_filter(signal, low_f, high_f, sample_rate, order=9):
    """perform a butterworth bandpass filter on a discrete time signal
    using scipy.signal's butter and solfiltfilt (phase-preserving version of sosfilt)
    
    signal: discrete time signal (audio samples, list of float)
    low_f: -3db point (?) for highpass filter (Hz)
    high_f: -3db point (?) for highpass filter (Hz)
    sample_rate: samples per second (Hz)
    order=9: higher values -> steeper dropoff
    
    return: filtered time signal 
    """
    sos = butter_bandpass(low_f, high_f, sample_rate, order=order)
    return sosfiltfilt(sos, signal)

#move to Audio or Librosa scripts
def mixdown(files_to_mix,out_dir,mix_name,levels=None,verbose=0,create_txt_file=True):
    """mix all files listed in file_to_mix into destination file as an mp3
    
    levels: optionally provide the ratios of each file's output amplitude in the mix as a list
    
    we will generate a text file for each one listing the origin files """
    
    levels_string = ''
    if levels is not None:
        if len(levels) != len(files_to_mix):
            raise ValueError("Length of levels must match length of files_to_mix")
        levels_string = f':weights="{" ".join([str(l) for l in levels])}"'
    
    inputs = ' -i '.join(files_to_mix)
    n_in = len(files_to_mix)
    overwrite = '-y' #-n to not overwrite
    
    out_file = out_dir + mix_name + '.mp3'
    
    cmd = f'ffmpeg {overwrite} -i {inputs} -filter_complex amix=inputs={n_in}:duration=longest{levels_string} {out_file}'
    
    if verbose>0:
        print(cmd)
        
    response =  run_command(cmd)

    if create_txt_file:
        metadata_file = out_dir + mix_name +'_info.txt'
        metadata = 'this file is a mixdown of: \n' + '\n'.join(files_to_mix)        
        with open(metadata_file,'w') as file:
            file.write(metadata)
    
    return response

#move to Audio or Librosa scripts
def create_mixdowns(in_files,out_dir,files_per_mix):
    """ take a set of audio files and layer them on top of eachother
    
    this method divides the set of files into files_per_mix sets
    and mixes (layers/sums) the audio: one file from each set
    
    """
    t = timer()
    print(f'saving mixdown files and logs to {out_dir}')
    print(f'logs contain list of origin files for each mix')

    if len(in_files)<1:
        print("didn't recieve any files!")

    random.shuffle(in_files)

    #split the files into n groups and overlap n files in each mixdown
    print(f'mixing {files_per_mix} files into each mixdown')

    n_files = len(in_files)
    print(f'total number of files: {n_files}')
    print(f'resulting mixdowns: {n_files//files_per_mix}')

    #split the file set into files_per_mix sets 
    idx = np.arange(0,n_files+1,n_files//files_per_mix)
    file_sets = np.array([ in_files[idx[i]:idx[i+1]] for i in range(files_per_mix)])

    #now we have separated lists of files to combine with eachother
    for i in range(len(file_sets[0])):
        #take the ith file from each set and mix into one
        files_to_mix = file_sets[:,i]
        mix_name = f'mix-{files_per_mix}-files_{i}'
        mixdown(files_to_mix,out_dir,mix_name)
        if i%50==0:
            print(f'completed {i} mixdowns of {len(file_sets[0])} in {timer()-t} seconds')
            
    print(f'copmleted mixdown task in {timer()-t} seconds \n')

#move to Audio or Librosa scripts
def convolve_file(in_file,out_file,ir_file,input_gain=1.0):
    """apply an impulse_response to a file using ffmpeg's afir convolution
    
    this makes the files 'sound as if' it were recorded
    in the location that the impulse response (ir_file) was recorded
    
    input_gain is a ratio for input sound's amplitude in (0,1)

    ir_file should be an audio file containing a short burst of noise
    recorded in a space whose acoustics are to be recreated """
    overwrite = '-y' #-n to not overwrite
    colvolve = f'-lavfi afir="{input_gain}:1:1:gn"'
    cmd = f'ffmpeg {overwrite} -i {in_file} -i {ir_file} {colvolve} {out_file}'
#     print(cmd)
    return run_command(cmd)

#move to Audio or Librosa scripts
def convolve_files(in_files,out_dir,ir_menu):
    """apply ffmpeg convolution to a set of files, 
    choosing the impulse response randomly from a list of paths (ir_menu)
    and saving modified files to out_dir/"""
    responses = []
    for i, in_file in enumerate(in_files):
        ir_file = random.choice(ir_menu)
        ir_applied = "".join(os.path.basename(ir_file)).split(".")[0:-1]
        out_file = f'{out_dir}{os.path.basename(in_file)}_{ir_applied}.mp3'

        response = convolve_file(in_file,out_file,ir_file)
        responses.append(response)

        if i%100 == 0:
            print(f'completed {i+1} convolutions of {len(split_files)}')
    print(f'completed {len(in_files)} convolutions')

#can act on an audio file and be moved into Audio class
def create_spectrum(path,bandpass_frequency_limits = None, start_end_times=None, plot=False):
    ''' given a file path, create a spectrum of energy across frequencies
    using a fast fourier transform
    
    if bandpass_frequency_limits is not none, the spectrum is band-passed
    to the frequencies [lowF, highF] which are given in hertz (Hz)
    
    if start_end_times is not None, we analyze only a part of the file
    start_end_time is given in seconds as [start, end]
    
    returns: fft, frequencies'''
    
    
    #load audio file with librosa, converting to mono if multi-channel, and down-sampling to 20.5kHz
    samples, sample_rate = load(
        path, mono=True, sr=20500, res_type="kaiser_fast"
    )
    
    #select the desired clip from the file using start_end_times
    if start_end_times is not None:
        start_sample = int(start_end_times[0]*sample_rate)
        end_sample = int(start_end_times[1]*sample_rate)
        samples = samples[start_sample:end_sample] #select samples in desired range
        #^we may be excluding one sample at the end but thats better than overshooting

    # Compute the fft (fast fourier transform) of the selected clip
    #this gives us the energy at different frequencies
    N = len(samples) #number of samples
    T = 1/sample_rate #sampling time step
    fft = scipy.fftpack.fft(samples) #compute fast fourier transform (fft)
    xf = np.fft.fftfreq(N, d=T) #the frequencies corresponding to fft[] elements
    
    
    #remove negative frequencies and scale magnitude by 2.0/N:
    fft = 2.0/N * fft[0:int(N/2)]
    frequencies = xf[0:int(N/2)]
    
    fft = np.abs(fft)

    #band pass filter the fft result
    if bandpass_frequency_limits is not None:
        lowF = bandpass_frequency_limits[0] #minimum frequency to keep
        highF = bandpass_frequency_limits[1] #maximum frequeny to keep
        filterVectorBool = np.logical_and(frequencies>=lowF,frequencies<=highF)
        fft = fft[filterVectorBool]
        frequencies = frequencies[filterVectorBool]
        
    if plot:
        plt.plot(frequencies, np.log(fft))
        plt.xlabel('frequency')
        plt.ylabel('fft')
#         plt.ylim([0,np.log(.02)])
        plt.xlim([50,10000])
        plt.title(os.path.basename(path))
        
        plt.show()
        
    return fft, frequencies

def clipping_detector(samples,threshold=0.6):
    ''' count the number of samples above a threshold value'''
    return len(list(filter(lambda x: x > threshold, samples)))

#helper for Tessa's silence detector, used for filtering xeno-canto and copied from crc: /ihome/sam/bmooreii/projects/opensoundscape/xeno-canto
#move to Audio module?
def window_energy(samples, nperseg = 256, noverlap = 128):
    '''
    Calculate audio energy with a sliding window
    
    Calculate the energy in an array of audio samples
    using a sliding window. Window includes nperseg
    samples per window and each window overlaps by
    noverlap samples. 
    
    Args:
        samples (np.ndarray): array of audio
            samples loaded using librosa.load
        
    
    '''
    def _energy(samples):
        return np.sum(samples**2)/len(samples)
    
    windowed = []
    skip = nperseg - noverlap
    for start in range(0, len(samples), skip):
        window_energy = _energy(samples[start : start + nperseg])
        windowed.append(window_energy)

    return windowed

#was detect_silence. Flipped outputs so that 0 is silent 1 is non-silent
def silence_filter(
    filename,
    smoothing_factor = 10,
    nperseg = 256,
    noverlap = 128,
    thresh = None
):
    '''
    Identify whether a file is silent
    
    Load samples from an mp3 file and identify
    whether or not it is likely to be silent.
    Silence is determined by finding the energy 
    in windowed regions of these samples, and
    normalizing the detected energy by the average
    energy level in the recording.
    
    If any windowed region has energy above the 
    threshold, returns a 0; else returns 1.
    
    Args:
        filename (str): file to inspect
        smoothing_factor (int): modifier
            to window nperseg
        nperseg: number of samples per window segment
        noverlap: number of samples to overlap
            each window segment
        thresh: threshold value (experimentally
            determined)
        
    Returns:
        0 if file contains no significant energy over bakcground
        1 if file contains significant energy over bakcground
    If thresh is None: returns net_energy over background noise
    '''
    try:
        samples, sr = load(filename,sr=None)
#     except NoBackendError:
#         return -1.0
    except RuntimeError:
        return -2.0
    except ZeroDivisionError:
        return -3.0
    except:
        return -4.0

    energy = window_energy(samples, nperseg*smoothing_factor, noverlap)
    norm_factor = np.mean(energy)
    net_energy = (energy - norm_factor)*100

    #the default of "None" for thresh will return the max value of ys
    if thresh is None:
        return np.max(net_energy)
    #if we pass a threshold (eg .05), we will return 0 or 1
    else: 
        return int(np.max(net_energy) > thresh)