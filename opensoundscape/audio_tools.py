"""
set of tools that work directly on audio files or samples

"""
from librosa import load 
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt 
import scipy
import os
from scipy.signal import butter, sosfiltfilt, sosfreqz

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

def butter_bandpass(low_f, high_f, sample_rate, order=9):
    """generate coefficients for bandpass filter"""
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

def clipping_detector(samples,threshold=0.6):
    ''' count the number of samples above a threshold value'''
    return len(list(filter(lambda x: x > threshold, samples)))

#helper for Tessa's silence detector, used for filtering xeno-canto and copied from crc: /ihome/sam/bmooreii/projects/opensoundscape/xeno-canto
#move to Audio module?
def window_energy(samples, window_len_samples = 256, overlap_len_samples = 128):
    '''
    Calculate audio energy with a sliding window
    
    Calculate the energy in an array of audio samples
    
    Args:
        samples (np.ndarray): array of audio
            samples loaded using librosa.load
        window_len_samples: samples per window
        overlap_len_samples: number of samples shared between consecutive windows
    
    Returns:
        list of energy level (float) for each window
    '''
    def _energy(samples):
        return np.sum(samples**2)/len(samples)
    
    windowed = []
    skip = window_len_samples - overlap_len_samples
    for start in range(0, len(samples), skip):
        window_energy = _energy(samples[start : start + window_len_samples])
        windowed.append(window_energy)

    return windowed

#Tessa's silence detector. was detect_silence(). Flipped outputs so that 0 is silent 1 is non-silent
def silence_filter(
    filename,
    smoothing_factor = 10,
    window_len_samples = 256,
    overlap_len_samples = 128,
    threshold = None
):
    '''
    Identify whether a file is silent (0) or not (1)
    
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
            to window_len_samples
        window_len_samples: number of samples per window segment
        overlap_len_samples: number of samples to overlap
            each window segment
        threshold: threshold value (experimentally
            determined)
        
    Returns:
        0 if file contains no significant energy over bakcground
        1 if file contains significant energy over bakcground
    If threshold is None: returns net_energy over background noise
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

    energy = window_energy(samples, window_len_samples*smoothing_factor, overlap_len_samples)
    norm_factor = np.mean(energy)
    net_energy = (energy - norm_factor)*100

    #the default of "None" for threshold will return the max value of ys
    if threshold is None:
        return np.max(net_energy)
    #if we pass a threshold (eg .05), we will return 0 or 1
    else: 
        return int(np.max(net_energy) > threshold)

def mixdown(files_to_mix,out_dir,mix_name,levels=None,duration='longest',verbose=0,create_txt_file=True):
    """mix all files listed in file_to_mix into destination file as an mp3
    
    levels: optionally provide the ratios of each file's output amplitude in the mix as a list
    duration='longest': duration of output mix, 'longest','shortest','first'
    
    we will generate a text file for each one listing the origin files """
    
    levels_string = ''
    if levels is not None:
        if len(levels) != len(files_to_mix):
            raise ValueError("Length of levels must match length of files_to_mix")
        levels_string = f':weights="{" ".join([str(l) for l in levels])}"'
    
    inputs = ' -i '.join(files_to_mix)
    n_in = len(files_to_mix)
    overwrite = '-y' #-n to not overwrite
    
    out_file = f'{out_dir}/{mix_name}.mp3'
    
    cmd = f'ffmpeg {overwrite} -i {inputs} -filter_complex amix=inputs={n_in}:duration={duration}{levels_string} {out_file}'
    
    if verbose>0:
        print(cmd)
        
    response =  run_command(cmd)

    if create_txt_file:
        metadata_file = out_dir + mix_name +'_info.txt'
        metadata = 'this file is a mixdown of: \n' + '\n'.join(files_to_mix)        
        with open(metadata_file,'w') as file:
            file.write(metadata)
    
    return response

def mixdown_with_delays(files_to_mix,destination,delays=None, levels=None,duration='first',verbose=0,create_txt_file=False):
    """use ffmpeg to mixdown a set of audio files, each starting at a specified time (padding beginnings with zeros)
    
    parameters:
        files_to_mix: list of audio file paths
        destination: path to save mixdown to
        delays=None: list of delays (how many seconds of zero-padding to add at beginning of each file)
        levels=None: optionally provide a list of relative levels (amplitudes) for each input
        duration='first': ffmpeg option for duration of output file: match duration of 'longest','shortest',or 'first' input file
        verbose=0: if >0, prints ffmpeg command and doesn't suppress ffmpeg output (command line output is returned from this function)
        create_txt_file=False: if True, also creates a second output file which lists all files that were included in the mixdown
        
    returns:
        ffmpeg command line output
    """
    
    #I'm including lots of comments because ffmpeg has confusing syntax
    #the trick with ffmpeg is to recognize the recurring syntax: [input][input2]function=option1=x:option2=y[output]
    #square-bracket syntax [s0] is basically a variable name, a place to put and get temporarily created objects
    #functions such as adelay (add delay at beginniing of audio) or amix (mix audio) are called with arguments
    #like: function=option1=x:option2=y
    #for instance: amix=inputs=2:duration=first
    #if we want to pass [0] (the first input) and [s1] (something we saved) to amix and save result to [mymix],
    #we would write: [0][s1]amix=inputs=2:duration=first[mymix]
    #yes, confusing, but that's why I've written a wrapper for it :) 
    
    n_inputs = len(files_to_mix)
    
    #format list of input files for ffmpeg (will look like -i file1.mp3 -i file2.wav)
    input_list = '-i ' + ' -i '.join(files_to_mix)
    
    #overwrite existing file by default? 'y' for yes, otherwise '' behavior is to not overwrite
    overwrite = '-y'
    
    #print all of ffmpegs messages?
    quiet_flag = '' if verbose>0 else ' -nostats -loglevel 0'
    
    options = f'{overwrite}{quiet_flag}'
        
    #if no delays are provided, they are all set to 0 (we could just skip it but syntax would be confusing)
    if delays is None:
        delays = np.zeros(n_inputs)
    
    #take each input {i}, delay it by delays[i], give the output a name
    #0 refers to first input file
    #here output of each adelay command is named s{i} (eg s0 for fist input)
    #will look like [0]adelay=0[s0];[1]adelay=1000[s1];
    delay_cmd = ''.join([ f'[{i}]adelay={delay}[s{i}];' for i, delay in enumerate(delays)] ) #for stereo, delay each chanel like adelay={delay}|{delay}
    
    #list of the outputs of adelay (these are the files we want to mixdown, so this is the input to amix)
    #will look like [s0][s1][s2]
    files_to_mix = ''.join([f'[s{i}]' for i in range(n_inputs)]) 
    
    #mixdown command 
    #take the files_to_mix, use amix to combine them into [mixdown]. Final duration is an option 'longest','shortest','first'
    #will look like [s0][s1]amix=inputs=2:duration=first[mixdown]
    mixdown_result = '[mixdown]' #think of this as a variable name for the result of amix
    mix_cmd = f'{files_to_mix}amix=inputs={n_inputs}:duration={duration}{mixdown_result}'

    #compile the full ffmpeg command: 
    #take inputs, apply delays, mix them together, and save to destination
    cmd = f'ffmpeg {options} {input_list} -filter_complex "{delay_cmd}{mix_cmd}" -map {mixdown_result} {destination}'
    
    if verbose>0:
        print(cmd)
        
    if create_txt_file:
        #we write a list of all input files to this mix into a .txt file with same name as .mp3. file
        metadata_file = destination +'_info.txt'
        metadata = 'this file is a mixdown of: \n' + '\n'.join(files_to_mix)        
        with open(metadata_file,'w') as file:
            file.write(metadata)
    
    return run_command(cmd)

# #move to scripts
# def create_mixdowns(in_files,out_dir,files_per_mix):
#     """ take a set of audio files and layer them on top of eachother
    
#     this method divides the set of files into files_per_mix sets
#     and mixes (layers/sums) the audio: one file from each set
    
#     """
#     t = timer()
#     print(f'saving mixdown files and logs to {out_dir}')
#     print(f'logs contain list of origin files for each mix')

#     if len(in_files)<1:
#         print("didn't recieve any files!")

#     random.shuffle(in_files)

#     #split the files into n groups and overlap n files in each mixdown
#     print(f'mixing {files_per_mix} files into each mixdown')

#     n_files = len(in_files)
#     print(f'total number of files: {n_files}')
#     print(f'resulting mixdowns: {n_files//files_per_mix}')

#     #split the file set into files_per_mix sets 
#     idx = np.arange(0,n_files+1,n_files//files_per_mix)
#     file_sets = np.array([ in_files[idx[i]:idx[i+1]] for i in range(files_per_mix)])

#     #now we have separated lists of files to combine with eachother
#     for i in range(len(file_sets[0])):
#         #take the ith file from each set and mix into one
#         files_to_mix = file_sets[:,i]
#         mix_name = f'mix-{files_per_mix}-files_{i}'
#         mixdown(files_to_mix,out_dir,mix_name)
#         if i%50==0:
#             print(f'completed {i} mixdowns of {len(file_sets[0])} in {timer()-t} seconds')
            
#     print(f'copmleted mixdown task in {timer()-t} seconds \n')


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

#move to scripts
# def convolve_files(in_files,out_dir,ir_menu):
#     """apply ffmpeg convolution to a set of files, 
#     choosing the impulse response randomly from a list of paths (ir_menu)
#     and saving modified files to out_dir/"""
#     responses = []
#     for i, in_file in enumerate(in_files):
#         ir_file = random.choice(ir_menu)
#         ir_applied = "".join(os.path.basename(ir_file)).split(".")[0:-1]
#         out_file = f'{out_dir}{os.path.basename(in_file)}_{ir_applied}.mp3'

#         response = convolve_file(in_file,out_file,ir_file)
#         responses.append(response)

#         if i%100 == 0:
#             print(f'completed {i+1} convolutions of {len(split_files)}')
#     print(f'completed {len(in_files)} convolutions')



