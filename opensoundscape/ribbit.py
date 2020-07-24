""" Detect periodic vocalizations with RIBBIT

This module provides functionality to search audio for periodically fluctuating vocalizations.
"""

import librosa
from scipy import signal
from librosa import load
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from time import time as timer
import os

# local imports
from opensoundscape.helpers import isNan, bound
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram


def calculate_pulse_score(
    amplitude, amplitude_sample_rate, pulse_rate_range, plot=False, nfft=1024
):
    """Search for amplitude pulsing in an audio signal in a range of pulse repetition rates (PRR)
    
    scores an audio amplitude signal by highest value of power spectral density in the PRR range
    
    Inputs:
        amlpitude: a time series of the audio signal's amplitude (for instance a smoothed raw audio signal) 
        amplitude_sample_rate: sample rate in Hz of amplitude signal, normally ~20-200 Hz
        pulse_rate_range: [min, max] values for amplitude modulation in Hz
        plot=False: if True, creates a plot visualizing the power spectral density
        nfft=1024: controls the resolution of the power spectral density (see scipy.signal.welch)
        
    Outputs: 
        pulse rate score for this audio segment (float) 
    """

    # input validation
    if len(amplitude) < 1:  # what is the minimum signal length?
        raise ValueError("amplitude does not have length > 0")

    # calculate amplitude modulation spectral density
    f, psd = signal.welch(amplitude, fs=amplitude_sample_rate, nfft=nfft)

    # look for the highest peak of power spectral density within pulse_rate_range
    min_rate = pulse_rate_range[0]
    max_rate = pulse_rate_range[1]
    psd_bandpassed = [psd[i] for i in range(len(psd)) if min_rate < f[i] < max_rate]

    if len(psd_bandpassed) < 1:
        return 0

    max_psd = np.max(psd_bandpassed)

    if plot:  # show where on the plot we are looking for a peak with vetical lines
        # check if a matplotlib backend is available
        import os

        if os.environ.get("MPLBACKEND") is None:
            import warnings

            warnings.warn("MPLBACKEND is 'None' in os.environ. Skipping plot.")
        else:
            from matplotlib import pyplot as plt
            from time import time

            # print(f"peak freq: {'{:.4f}'.format(f[np.argmax(psd)])}")
            plt.plot(f, psd)
            plt.plot([pulse_rate_range[0], pulse_rate_range[0]], [0, max_psd])
            plt.plot([pulse_rate_range[1], pulse_rate_range[1]], [0, max_psd])
            plt.title("RIBBIT: power spectral density")
            plt.xlabel("pulse rate (pulses/sec)")
            plt.ylabel("power spectral density")
            plt.show()

    return max_psd


def ribbit(
    spectrogram, signal_band, pulse_rate_range, window_len, noise_bands=None, plot=False
):
    """Run RIBBIT detector to search for periodic calls in audio
    
    This tool searches for periodic energy fluctuations at specific repetition rates and frequencies. 
    
    Inputs:
        spectrogram: opensoundscape.Spectrogram object of an audio file
        signal_band: [min, max] frequency range of the target species, in Hz
        pulse_rate_range: [min,max] pulses per second for the target species
        windo_len: the length of audio (in seconds) to analyze at one time
                    - one RIBBIT score is produced for each window
        noise_bands: list of frequency bands to subtract from the desired signal_band
                    For instance: [ [min1,max1] , [min2,max2] ]
                    - if `None`, no noise bands are used
                    - default: None
        plot=False : if True, plot the power spectral density for each window
    
    Outputs:
        array of pulse_score: pulse score (float) for each time window
        array of time: start time of each window
        
    Notes
    -----
    
    __PARAMETERS__ 
    RIBBIT requires the user to select a set of parameters that describe the target vocalization. Here is some detailed advice on how to use these parameters.

    **Signal Band:** The signal band is the frequency range where RIBBIT looks for the target species. It is best to pick a narrow signal band if possible, so that the model focuses on a specific part of the spectrogram and has less potential to include erronious sounds. 

    **Noise Bands:** Optionally, users can specify other frequency ranges called noise bands. Sounds in the `noise_bands` are _subtracted_ from the `signal_band`. Noise bands help the model filter out erronious sounds from the recordings, which could include confusion species, background noise, and popping/clicking of the microphone due to rain, wind, or digital errors. It's usually good to include one noise band for very low frequencies -- this specifically eliminates popping and clicking from being registered as a vocalization. It's also good to specify noise bands that target confusion species. Another approach is to specify two narrow `noise_bands` that are directly above and below the `signal_band`. 

    **Pulse Rate Range:** This parameters specifies the minimum and maximum pulse rate (the number of pulses per second, also known as pulse repetition rate) RIBBIT should look for to find the focal species. For example, choosing `pulse_rate_range = [10, 20]` means that RIBBIT should look for pulses no slower than 10 pulses per second and no faster than 20 pulses per second. 

    **Window Length:** This parameter tells RIBBIT how many seconds of audio to analyze at one time. Generally, you should choose a `window_length` that is similar to the length of the target species vocalization, or a little bit longer. For very slowly pulsing vocalizations, choose a longer window so that at least 5 pulses can occur in one window (0.5 pulses per second -> 10 second window). Typical values for `window_length` are 1 to 10 seconds. 

    **Plot:** We can choose to show the power spectrum of pulse repetition rate for each window by setting `plot=True`. The default is not to show these plots (`plot=False`).
    
    __ALGORITHM__
    This is the procedure RIBBIT follows:
    divide the audio into segments of length window_len
    for each clip:
        calculate time series of energy in signal band (signal_band) and subtract noise band energies (noise_bands)
        calculate power spectral density of the amplitude time series
        score the file based on the maximum value of power-spectral-density in the pulse rate range
    
    """

    # Make a 1d amplitude signal in a frequency range, subtracting energy in noise bands
    amplitude = spectrogram.net_amplitude(signal_band, noise_bands)

    # next we split the spec into "windows" to analyze separately: (no overlap for now)
    sample_frequency_of_spec = (len(spectrogram.times) - 1) / (
        spectrogram.times[-1] - spectrogram.times[0]
    )  # in Hz, ie delta-t between consecutive pixels
    n_samples_per_window = int(window_len * sample_frequency_of_spec)
    signal_len = len(amplitude)

    start_sample = 0
    pulse_scores = []
    window_start_times = []

    # step through the file, analyzing in pieces that are window_len long, saving scores and start times for each window
    while start_sample + n_samples_per_window < signal_len - 1:
        ta = timer()

        end_sample = start_sample + n_samples_per_window
        if end_sample < signal_len:
            window = amplitude[start_sample:end_sample]
        else:
            final_start_sample = max(0, signal_len - n_samples_per_window)
            window = amplitude[final_start_sample:signal_len]

        if plot:
            print(
                f"window: {'{:.4f}'.format(start_sample/sample_frequency_of_spec)} sec to {'{:.4f}'.format(end_sample/sample_frequency_of_spec)} sec"
            )
        # Make psd (Power spectral density or power spectrum of x) and find max
        pulse_score = calculate_pulse_score(
            window, sample_frequency_of_spec, pulse_rate_range, plot
        )

        # save results
        pulse_scores.append(pulse_score)
        window_start_times.append(start_sample / sample_frequency_of_spec)

        # update start_sample
        start_sample = (
            end_sample
        )  # end sample was excluded so use it as first sample in next window

    return pulse_scores, window_start_times


def pulse_finder_species_set(spec, species_df, window_len="from_df", plot=False):
    """ perform windowed pulse finding (ribbit) on one file for each species in a set

    Args:
        spec: opensoundscape.Spectrogram object
        species_df: a dataframe describing species by their pulsed calls. 
            columns: species | pulse_rate_low (Hz)| pulse_rate_high (Hz) | low_f (Hz)| high_f (Hz)| reject_low (Hz)| reject_high (Hz) | 
                    window_length (sec) (optional) | reject_low2 (opt) | reject_high2 |
        window_len: length of analysis window, in seconds. 
                    Or 'from_df' (default): read from dataframe. 
                    or 'dynamic': adjust window size based on pulse_rate
    
    Returns: 
        the same dataframe with a "score" (max score) column and "time_of_score" column
    """

    species_df = species_df.copy()
    species_df = species_df.set_index(species_df.columns[0], drop=True)

    species_df["score"] = [[] for i in range(len(species_df))]
    species_df["t"] = [[] for i in range(len(species_df))]
    species_df["max_score"] = [np.nan for i in range(len(species_df))]

    for i, row in species_df.iterrows():

        # we can't analyze pulse rates of 0 or NaN
        if isNan(row.pulse_rate_low) or row.pulse_rate_low == 0:
            # cannot analyze
            continue

        pulse_rate_range = [row.pulse_rate_low, row.pulse_rate_high]

        if window_len == "from_df":
            window_len = row.window_length
        elif window_len == "dynamic":
            # dynamically choose the window length based on the species pulse-rate
            # try to capture ~4-10 pulses
            min_len = 0.5  # sec
            max_len = 10  # sec
            target_n_pulses = 5
            window_len = bound(
                target_n_pulses / pulse_rate_range[0], [min_len, max_len]
            )
        # otherwise, use the numerical value provided for window length

        signal_band = [row.low_f, row.high_f]  # changed from low_f, high_f

        noise_bands = None
        if not isNan(row.reject_low):
            noise_bands = [[row.reject_low, row.reject_high]]
            if "reject_low2" in species_df.columns and not isNan(row.reject_low2):
                noise_bands.append([row.reject_low2, row.reject_high2])

        # score this species for each window using ribbit
        if plot:
            print(f"{row.name}")
        pulse_scores, window_start_times = ribbit(
            spec, signal_band, pulse_rate_range, window_len, noise_bands, plot
        )

        # add the scores to the species df
        species_df.at[i, "score"] = pulse_scores
        species_df.at[i, "t"] = window_start_times
        species_df.at[i, "max_score"] = (
            max(pulse_scores) if len(pulse_scores) > 0 else np.nan
        )
        species_df.at[i, "time_of_max_score"] = (
            window_start_times[np.argmax(pulse_scores)]
            if len(pulse_scores) > 0
            else np.nan
        )

    return species_df


def summarize_top_scores(audio_files, list_of_result_dfs, scale_factor=1.0):
    """ find the highest score for each file and each species, and put them in a dataframe 
    
    Note: this function expects that the first column of the results_df contains species names 
    
    Args:
        audio_files: a list of file paths
        list_of_result_dfs: a list of pandas DataFrames generated by ribbit_species_set()
        scale_factor=1.0: optionally multiply all output values by a constant value
        
    Returns: 
        a dataframe summarizing the highest score for each species in each file
        
    """
    if len(audio_files) != len(list_of_result_dfs):
        raise ValueError(
            "The length of audio_files must match the length of list_of_results_dfs"
        )

    import pandas as pd

    top_species_scores_df = pd.DataFrame(
        index=audio_files, columns=list_of_result_dfs[0].iloc[:, 0]
    )
    top_species_scores_df.index.name = "file"

    all_results_dfs = []
    for i, f in enumerate(audio_files):
        results_df = list_of_result_dfs[i]
        results_df = results_df.set_index(results_df.columns[0])
        for sp in results_df.index:
            top_species_scores_df.at[f, sp] = (
                results_df.at[sp, "max_score"] * scale_factor
            )

    return top_species_scores_df
