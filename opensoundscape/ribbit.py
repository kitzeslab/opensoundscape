""" Detect periodic vocalizations with RIBBIT

This module provides functionality to search audio for periodically fluctuating vocalizations.
"""

from scipy import signal
import numpy as np

# local imports
from opensoundscape.helpers import isNan, bound, generate_clip_times_df


def calculate_pulse_score(
    amplitude, amplitude_sample_rate, pulse_rate_range, plot=False, nfft=1024
):
    """Search for amplitude pulsing in an audio signal in a range of pulse repetition rates (PRR)

    scores an audio amplitude signal by highest value of power spectral density in the PRR range

    Args:
        amplitude: a time series of the audio signal's amplitude (for instance a smoothed raw audio signal)
        amplitude_sample_rate: sample rate in Hz of amplitude signal, normally ~20-200 Hz
        pulse_rate_range: [min, max] values for amplitude modulation in Hz
        plot=False: if True, creates a plot visualizing the power spectral density
        nfft=1024: controls the resolution of the power spectral density (see scipy.signal.welch)

    Returns:
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
    spectrogram,
    signal_band,
    pulse_rate_range,
    clip_duration,
    clip_overlap=0,
    final_clip=None,
    noise_bands=None,
    plot=False,
):
    """Run RIBBIT detector to search for periodic calls in audio

    This tool searches for periodic energy fluctuations at specific repetition rates and frequencies.


    Args:
        spectrogram: opensoundscape.Spectrogram object of an audio file
        signal_band: [min, max] frequency range of the target species, in Hz
        pulse_rate_range: [min,max] pulses per second for the target species
        clip_duration: the length of audio (in seconds) to analyze at one time
            - each clip is analyzed independently and recieves a ribbit score
        clip_overlap (float):   overlap between consecutive clips (sec)
        final_clip (str):       behavior if final clip is less than clip_duration
            seconds long. By default, discards remaining audio if less than
            clip_duration seconds long [default: None].
            Options:
            - None:         Discard the remainder (do not make a clip)
            - "remainder":  Use only remainder of Audio (final clip will be shorter than clip_duration)
            - "full":       Increase overlap with previous clip to yield a clip with clip_duration length
            Note that the "extend" option is not supported for RIBBIT.

        noise_bands: list of frequency ranges to subtract from the signal_band
            For instance: [ [min1,max1] , [min2,max2] ]
            - if `None`, no noise bands are used
            - default: None
        plot=False: if True, plot the power spectral density for each clip

    Returns:
        DataFrame of index=('start_time','end_time'), columns=['score'],
        with a row for each clip.

    Notes
    -----

    __PARAMETERS__
    RIBBIT requires the user to select a set of parameters that describe the target vocalization. Here is some detailed advice on how to use these parameters.

    **Signal Band:** The signal band is the frequency range where RIBBIT looks for the target species. It is best to pick a narrow signal band if possible, so that the model focuses on a specific part of the spectrogram and has less potential to include erronious sounds.

    **Noise Bands:** Optionally, users can specify other frequency ranges called noise bands. Sounds in the `noise_bands` are _subtracted_ from the `signal_band`. Noise bands help the model filter out erronious sounds from the recordings, which could include confusion species, background noise, and popping/clicking of the microphone due to rain, wind, or digital errors. It's usually good to include one noise band for very low frequencies -- this specifically eliminates popping and clicking from being registered as a vocalization. It's also good to specify noise bands that target confusion species. Another approach is to specify two narrow `noise_bands` that are directly above and below the `signal_band`.

    **Pulse Rate Range:** This parameters specifies the minimum and maximum pulse rate (the number of pulses per second, also known as pulse repetition rate) RIBBIT should look for to find the focal species. For example, choosing `pulse_rate_range = [10, 20]` means that RIBBIT should look for pulses no slower than 10 pulses per second and no faster than 20 pulses per second.

    **Clip Duration:** The `clip_duration` parameter tells RIBBIT how many seconds of audio to analyze at one time. Generally, you should choose a `clip_length` that is similar to the length of the target species vocalization, or a little bit longer. For very slowly pulsing vocalizations, choose a longer window so that at least 5 pulses can occur in one window (0.5 pulses per second -> 10 second window). Typical values for are 0.3 to 10 seconds.
    Also, `clip_overlap` can be used for overlap between sequential clips. This
    is more computationally expensive but will be more likely to center a target
    sound in the clip (with zero overlap, the target sound may be split up between
    adjacent clips).

    **Plot:** We can choose to show the power spectrum of pulse repetition rate for each window by setting `plot=True`. The default is not to show these plots (`plot=False`).

    __ALGORITHM__
    This is the procedure RIBBIT follows:
    divide the audio into segments of length clip_duration
    for each clip:
        calculate time series of energy in signal band (signal_band) and subtract noise band energies (noise_bands)
        calculate power spectral density of the amplitude time series
        score the file based on the maximum value of power spectral density in the pulse rate range

    """
    if final_clip == "extend":
        raise ValuseError(
            "final_clip='extend' is not supported for RIBBIT. "
            "consider using 'remainder'."
        )

    # Make a 1d amplitude signal from signal_band & subtract amplitude from noise bands
    amplitude = np.array(spectrogram.net_amplitude(signal_band, noise_bands))
    time = spectrogram.times
    # we calculate the sample rate of the amplitude signal using the difference
    # in time between columns of the Spectrogram
    sample_rate = 1 / spectrogram.window_step()

    # determine the start and end times of each clip to analyze
    clip_df = generate_clip_times_df(
        full_duration=spectrogram.duration(),
        clip_duration=clip_duration,
        clip_overlap=clip_overlap,
        final_clip=final_clip,
    )
    clip_df["score"] = np.nan

    # analyze each clip and save scores in the clip_df
    for i, row in clip_df.iterrows():

        # extract the amplitude signal for this clip
        window = amplitude[(time >= row["start_time"]) & (time < row["end_time"])]

        if plot:
            print(f"window: {row['start_time']} to {row['end_time']} sec")

        # calculate score for this clip:
        # - make psd (Power spectral density or power spectrum of amplitude)
        # - find max value in the pulse_rate_range
        clip_df.at[i, "score"] = calculate_pulse_score(
            window, sample_rate, pulse_rate_range, plot=plot
        )

    return clip_df
