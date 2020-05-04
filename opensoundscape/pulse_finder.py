import librosa
from scipy import signal
from librosa import load
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from time import time as timer
import os

#local imports
from opensoundscape.helpers import isNan, bound
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

def calculate_pulse_score(amplitude, sample_frequency_of_spec, pulserate_range, plot=False,nfft=1024):#1024
    """score an audio amplitude signal by finding the maximum value of the power spectral density inside a range of pulse rates"""
    
    #calculate amplitude modulation spectral density
    f, pxx = signal.welch(amplitude, fs=sample_frequency_of_spec, nfft=nfft)
    
    #look for the highest peak of power spectral density within pulserate_range    
    min_rate = pulserate_range[0]
    max_rate = pulserate_range[1]
    pxx_bandpassed = [pxx[i] for i in range(len(pxx)) if min_rate<f[i]<max_rate]
    
    if len(pxx_bandpassed)<1:
        return 0
    
    max_pxx = np.max(pxx_bandpassed)
    
    if(plot): #show where on the plot we are looking for a peak with vetical lines
        #check if a matplotlib backend is available 
        import os
        if os.environ.get('MPLBACKEND') is None:
            import warnings
            warnings.warn("MPLBACKEND is 'None' in os.environ. Skipping plot.")
        else:
            from matplotlib import pyplot as plt
            from time import time 
            print(f"peak freq: {f[np.argmax(pxx)]}")
            plt.plot(f,pxx)
#             plt.xlim([0,100])
#             plt.ylim([0,.001])
            plt.plot([pulserate_range[0],pulserate_range[0]],[0,max_pxx])
            plt.plot([pulserate_range[1],pulserate_range[1]],[0,max_pxx])
            plt.show()

    return max_pxx

def pulse_finder(spectrogram, freq_range, pulserate_range, window_len, rejection_bands=None, plot=False):
    '''pulse rate method for repeated pulse calls: 
    look in a box of frequency and pulse-rate, score on amplitude of fft of smooth signal amplitude
    divides the audio into segments of length window_len
    
    args:
        spectrogram: opensoundscape.Spectrogram object 
        freq_range: range to bandpass the spectrogram, in Hz
        pulserate_range: how many pulses per second? (where to look in the fft of the smoothed-amplitude), in Hz
        rejection_bands: list of frequency bands to subtract from the desired freq_range
        plot=False : if True, plot figures
    
    returns:
        array of pulse_score: max(fft) in the 2-d constraints
        array of time: start time of each window
    '''
    
    # Make a 1d amplitude signal in a frequency range, subtracting energy in rejection bands
    amplitude = spectrogram.net_amplitude(freq_range,rejection_bands)
    
    #next we split the spec into "windows" to analyze separately: (no overlap for now)
    sample_frequency_of_spec = (len(spectrogram.times)-1)/(spectrogram.times[-1]-spectrogram.times[0]) #in Hz, ie delta-t between consecutive pixels
    n_samples_per_window = int(window_len * sample_frequency_of_spec ) 
    signal_len = len(amplitude)
    
    start_sample = 0
    pulse_scores = []
    window_start_times = []
    
    #step through the file, analyzing in pieces that are window_len long, saving scores and start times for each window
    while start_sample + n_samples_per_window < signal_len -1:
        ta = timer()
        
        end_sample = start_sample + n_samples_per_window
        if end_sample < signal_len:
            window = amplitude[start_sample : end_sample ]
        else:
            final_start_sample = max(0, signal_len-n_samples_per_window)
            window = amplitude[final_start_sample : signal_len]
            
        # Make Pxx (Power spectral density or power spectrum of x) and find max
        pulse_score = calculate_pulse_score(window,sample_frequency_of_spec,pulserate_range, plot)

        #save results
        pulse_scores.append(pulse_score)
        window_start_times.append(start_sample/sample_frequency_of_spec)
        
        #update start_sample
        start_sample = end_sample #end sample was excluded so use it as first sample in next window
        
    return pulse_scores, window_start_times

# the following functions are wrappers/workflows/recipies that make it easy to run pulse_finder on multiple files for multiple species. 
def pulse_finder_file(file, freq_range, pulserate_range, window_len, rejection_bands=None, plot=False):
    '''pulse rate method, breaking file into "windows" (segments of audio file) of length window_len (seconds): 
    look in a range of frequencies and pulse-rates, score is the amplitude of fft of net amplitude = [the amplitude in frequency band] 
    minus [the amplitude in rejection bands]
    
    args:
        file: path to an audio file to search for pulsing call 
        freq_range: range to bandpass the spectrogram, in Hz
        pulserate_range: how many pulses per second? (where to look in the fft of the smoothed-amplitude), in Hz
        plot=False : if True, plot figures
        
    returns:
        array of pulse_score: max(fft) in the 2-d constraints
        array of time: start time of each window
        '''
    #make spectrogram from file path
    audio = Audio(file)
    spec = Spectrogram.from_audio(audio)
    
    pulse_scores, window_start_times = pulse_finder(spec, freq_range, pulserate_range, window_len, rejection_bands, plot)

    return pulse_scores, atom_start_times

def pulse_finder_species_set(spec, species_df, window_len = 'from_df', plot=False):
    """ perform windowed pulse finding on one file for each species in a set

    parameters:
    spec: opensoundscape.Spectrogram object
    species_df: a dataframe describing species by their pulsed calls. columns: species | pulse_rate_low (Hz)| pulse_rate_high (Hz) | low_f (Hz)| high_f (Hz)| reject_low (Hz)| reject_high (Hz) | window_length (sec) (optional) | reject_low2 (opt) | reject_high2 |
    window_len: length of analysis window, in seconds. Or 'from_df' (default): read from dataframe. or 'dynamic': adjust window size based on pulse_rate
    
    returns: the same dataframe with a "score" (max score) column and "time_of_score" column
    """
    
    species_df = species_df.copy()
    species_df = species_df.set_index(species_df.columns[0],drop=True)
    
    species_df['score'] = [ [] for i in range(len(species_df))]
    species_df['t'] = [ [] for i in range(len(species_df))]
    species_df['max_score'] = [np.nan for i in range(len(species_df))]
    
    for i, row in species_df.iterrows(): 
                
        #we can't analyze pulse rates of 0 or NaN
        if isNan(row.pulse_rate_low) or row.pulse_rate_low == 0:
            #cannot analyze
            continue
        
        pulserate_range = [row.pulse_rate_low,row.pulse_rate_high]
        
        if window_len == "from_df":
            window_len = row.window_length
        elif window_len == 'dynamic':
            #dynamically choose the window length based on the species pulse-rate
            #try to capture ~4-10 pulses
            min_len = .5 #sec
            max_len = 10 #sec
            target_n_pulses = 5
            window_len = bound( target_n_pulses/pulserate_range[0], [min_len,max_len] )
        #otherwise, use the numerical value provided for window length
        
        freq_range = [row.low_f, row.high_f] #changed from low_f, high_f

        rejection_bands = None
        if not isNan(row.reject_low):
            rejection_bands = [[row.reject_low,row.reject_high]]
            if "reject_low2" in species_df.columns and not isNan(row.reject_low2):
                rejection_bands.append([row.reject_low2,row.reject_high2])
                    
        #score this species for each window using pulse_finder
        if plot: print(f'{row.name}')
        pulse_scores, window_start_times = pulse_finder(spec, freq_range, pulserate_range, window_len, rejection_bands, plot)
        
        #add the scores to the species df
        species_df.at[i,'score'] = pulse_scores
        species_df.at[i,'t'] = window_start_times
        species_df.at[i,'max_score'] = max(pulse_scores)
        species_df.at[i,'time_of_max_score'] = window_start_times[np.argmax(pulse_scores)]
        
    return species_df

def summarize_top_scores(audio_files, list_of_result_dfs, scale_factor=1.0):
    ''' find the highest score for each file and each species, and put them in a dataframe 
    
    Note: this function expects that the first column of the results_df contains species names 
    
    use scale_factor to multiply all scores by a constant value '''
    if len(audio_files) != len(list_of_result_dfs):
        raise ValueError('The length of audio_files must match the length of list_of_results_dfs')
    
    import pandas as pd
    
    top_species_scores_df = pd.DataFrame(index=audio_files,columns = list_of_result_dfs[0].iloc[:,0])
    top_species_scores_df.index.name = 'file'

    all_results_dfs = []
    for i,f in enumerate(audio_files):
        results_df = list_of_result_dfs[i]
        results_df = results_df.set_index(results_df.columns[0])
        for sp in results_df.index:
            top_species_scores_df.at[f,sp]=results_df.at[sp,'max_score']*scale_factor   
    
    return top_species_scores_df