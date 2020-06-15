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
    
    Args:
        amlpitude: a time series of the audio signal's amplitude (for instance a smoothed raw audio signal) 
        amplitude_sample_rate: sample rate in Hz of amplitude signal, normally ~20-200 Hz
        pulse_rate_range: [min, max] values for amplitude modulation in Hz
        plot=False: if True, creates a plot visualizing the power spectral density
        nfft=1024: controls the resolution of the power spectral density (see scipy.signal.welch)
        
    Returns: pulse rate score for this audio segment (float) """

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

            print(f"peak freq: {f[np.argmax(psd)]}")
            plt.plot(f, psd)
            plt.plot([pulse_rate_range[0], pulse_rate_range[0]], [0, max_psd])
            plt.plot([pulse_rate_range[1], pulse_rate_range[1]], [0, max_psd])
            plt.show()

    return max_psd


def pulse_finder(
    spectrogram,
    freq_range,
    pulse_rate_range,
    window_len,
    rejection_bands=None,
    plot=False,
):
    """Run pulse rate method to search for pulsing calls in audio
    
    algorithm:
    divide the audio into segments of length window_len
    for each clip:
        calculate time series of energy in signal band (freq_range) and subtract noise band energies (rejection_bands)
        calculate power spectral density of the amplitude time series
        score the file based on the maximum value of power-spectral-density in the PRR range
    
    Args:
        spectrogram: opensoundscape.Spectrogram object of an audio file
        freq_range: range to bandpass the spectrogram, in Hz
        pulse_rate_range: how many pulses per second? (where to look in the fft of the smoothed-amplitude), in Hz
        rejection_bands: list of frequency bands to subtract from the desired freq_range
        plot=False : if True, plot figures
    
    Returns:
        array of pulse_score: pulse score (float) for each time window
        array of time: start time of each window
    """

    # Make a 1d amplitude signal in a frequency range, subtracting energy in rejection bands
    amplitude = spectrogram.net_amplitude(freq_range, rejection_bands)

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
                f"window: {start_sample/sample_frequency_of_spec} sec to {end_sample/sample_frequency_of_spec} sec"
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


# # the following functions are wrappers/workflows/recipies that make it easy to run pulse_finder on multiple files for multiple species.
# def pulse_finder_file(
#     file, freq_range, pulse_rate_range, window_len, rejection_bands=None, plot=False
# ):
#     """a wrapper for pulse_finder with takes an audio file path as an argument

#     creates the audio object and spectrogram within the function

#     Args:
#         file: path to an audio file
#         freq_range: range to bandpass the spectrogram, in Hz
#         pulse_rate_range: how many pulses per second? (where to look in the fft of the smoothed-amplitude), in Hz
#         rejection_bands: list of frequency bands to subtract from the desired freq_range
#         plot=False : if True, plot figures

#     Returns:
#         array of pulse_score: pulse score (float) for each time window
#         array of time: start time of each window

#     """
#     # make spectrogram from file path
#     audio = Audio.from_file(file)
#     spec = Spectrogram.from_audio(audio)

#     pulse_scores, window_start_times = pulse_finder(
#         spec, freq_range, pulse_rate_range, window_len, rejection_bands, plot
#     )

#     return pulse_scores, window_start_times


def pulse_finder_species_set(spec, species_df, window_len="from_df", plot=False):
    """ perform windowed pulse finding on one file for each species in a set

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

        freq_range = [row.low_f, row.high_f]  # changed from low_f, high_f

        rejection_bands = None
        if not isNan(row.reject_low):
            rejection_bands = [[row.reject_low, row.reject_high]]
            if "reject_low2" in species_df.columns and not isNan(row.reject_low2):
                rejection_bands.append([row.reject_low2, row.reject_high2])

        # score this species for each window using pulse_finder
        if plot:
            print(f"{row.name}")
        pulse_scores, window_start_times = pulse_finder(
            spec, freq_range, pulse_rate_range, window_len, rejection_bands, plot
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
        list_of_result_dfs: a list of pandas DataFrames generated by pulse_finder_species_set()
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
