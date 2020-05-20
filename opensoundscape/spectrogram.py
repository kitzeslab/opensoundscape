#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np
from opensoundscape.audio import Audio
import warnings
import pickle


class Spectrogram:
    """ Immutable spectrogram container
    """

    __slots__ = (
        "frequencies",
        "times",
        "spectrogram",
    )  # , "overlap", "segment_length")

    def __init__(self, spectrogram, frequencies, times):
        if not isinstance(spectrogram, np.ndarray):
            raise TypeError(
                f"Spectrogram.spectrogram should be a np.ndarray [shape=(n, m)]. Got {spectrogram.__class__}"
            )
        if not isinstance(frequencies, np.ndarray):
            raise TypeError(
                f"Spectrogram.frequencies should be an np.ndarray [shape=(n,)]. Got {frequencies.__class__}"
            )
        if not isinstance(times, np.ndarray):
            raise TypeError(
                f"Spectrogram.times should be an np.ndarray [shape=(m,)]. Got {times.__class__}"
            )

        if spectrogram.ndim != 2:
            raise TypeError(
                f"spectrogram should be a np.ndarray [shape=(n, m)]. Got {spectrogram.shape}"
            )
        if frequencies.ndim != 1:
            raise TypeError(
                f"frequencies should be an np.ndarray [shape=(n,)]. Got {frequencies.shape}"
            )
        if times.ndim != 1:
            raise TypeError(
                f"times should be an np.ndarray [shape=(m,)]. Got {times.shape}"
            )

        if spectrogram.shape != (frequencies.shape[0], times.shape[0]):
            raise TypeError(
                f"Dimension mismatch, spectrogram.shape: {spectrogram.shape}, frequencies.shape: {frequencies.shape}, times.shape: {times.shape}"
            )

        super(Spectrogram, self).__setattr__("frequencies", frequencies)
        super(Spectrogram, self).__setattr__("times", times)
        super(Spectrogram, self).__setattr__("spectrogram", spectrogram)

    @classmethod
    def from_audio(
        cls,
        audio,
        window_type="hann",
        window_samples=512,
        overlap_samples=256,
        decibel_limits=(-100, -20),
    ):
        """
        create a Spectrogram object from an Audio object
        
        parameters:
            window_type="hann": see scipy.signal.spectrogram docs for description of window parameter
            window_samples=512: number of audio samples per spectrogram window (pixel)
            overlap_samples=256: number of samples shared by consecutive windows
            decibels=True: convert the spectrogram values to decibelss (dB)
            decibel_limits = (-100,-20) : limit the dB values to (min,max) (lower values set to min, higher values set to max)
            
        returns:
            opensoundscape.spectrogram.Spectrogram object
        """
        if not isinstance(audio, Audio):
            raise TypeError("Class method expects Audio class as input")

        frequencies, times, spectrogram = signal.spectrogram(
            audio.samples,
            audio.sample_rate,
            window=window_type,
            nperseg=window_samples,
            noverlap=overlap_samples,
            scaling="spectrum",
        )

        # convert to decibels
        spectrogram = 10 * np.log10(spectrogram)

        # limit the decibel range (-100 to -20 dB by default)
        # values below lower limit set to lower limit, values above upper limit set to uper limit
        min_db, max_db = decibel_limits
        spectrogram[spectrogram > max_db] = max_db
        spectrogram[spectrogram < min_db] = min_db

        return cls(spectrogram, frequencies, times)

    def __setattr__(self, name, value):
        raise AttributeError("Spectrogram's cannot be modified")

    def __repr__(self):
        return f"<Spectrogram(spectrogram={self.spectrogram.shape}, frequencies={self.frequencies.shape}, times={self.times.shape})>"

    def min_max_scale(self, feature_range=(0, 1)):
        """ Apply a min-max filter
        """

        if len(feature_range) != 2:
            raise AttributeError(
                "Error: `feature_range` doesn't look like a 2-element tuple?"
            )
        if feature_range[1] < feature_range[0]:
            raise AttributeError("Error: `feature_range` isn't increasing?")

        spect_min = self.spectrogram.min()
        spect_max = self.spectrogram.min()
        scale_factor = (feature_range[1] - feature_range[0]) / (spect_max - spect_min)
        return Spectrogram(
            scale_factor * (self.spectrogram - spect_min) + feature_range[0],
            self.frequencies,
            self.times,
        )

    def limit_db_range(self, min_db=-100, max_db=-20):
        """ Limit the decibel values of the spectrogram to range from min_db to max_db
            
            values less than min_db are set to min_db
            values greater than max_db are set to max_db
            
            similar to Audacity's gain and range parameters
        """
        _spec = self.spectrogram

        _spec[_spec > max_db] = max_db
        _spec[_spec < min_db] = min_db

        return Spectrogram(_spec, self.frequencies, self.times)

    def bandpass(self, min_f, max_f):
        """ extract a frequency band from a spectrogram

        params:
        min_f: low frequency in Hz for bandpass
        high_f: high frequency in Hz for bandpass
        
        returns:
        bandpassed spectrogram object

        """

        # find indices of the frequencies in spec_freq closest to min_f and max_f
        lowest_index = np.abs(self.frequencies - min_f).argmin()
        highest_index = np.abs(self.frequencies - max_f).argmin()

        # take slices of the spectrogram and spec_freq that fall within desired range
        return Spectrogram(
            self.spectrogram[lowest_index:highest_index, :],
            self.frequencies[lowest_index:highest_index],
            self.times,
        )

    def trim(self, start_time, end_time):
        """ extract a time segment from a spectrogram

        params:
        start_time: in seconds
        end_time: in seconds

        returns:
        spectrogram object from extracted time segment
        
        """

        # find indices of the times in self.times closest to min_t and max_t
        lowest_index = np.abs(self.times - start_time).argmin()
        highest_index = np.abs(self.times - end_time).argmin()

        # take slices of the spectrogram and spec_freq that fall within desired range
        return Spectrogram(
            self.spectrogram[:, lowest_index:highest_index],
            self.frequencies,
            self.times[lowest_index:highest_index],
        )

    def plot(self, inline=True, fname=None, show_colorbar=False):
        """Plot the spectrogram with pyplot. 
        
        Parameters:
            inline=True: 
            fname=None: specify a string path to save the plot to (ending in .png/.pdf)
            show_colorbar: include image legend colorbar from pyplot
        """
        from matplotlib import pyplot as plt

        plt.pcolormesh(self.times, self.frequencies, self.spectrogram)
        plt.xlabel("time (sec)")
        plt.ylabel("frequency (Hz)")
        if show_colorbar:
            plt.colorbar()

        # if fname is not None, save to file path fname
        if fname:
            plt.savefig(fname)

        # if not saving to file, check if a matplotlib backend is available
        if inline:
            import os

            if os.environ.get("MPLBACKEND") is None:
                warnings.warn("MPLBACKEND is 'None' in os.environ. Skipping plot.")
            else:
                plt.show()

    def amplitude(
        self, freq_range=None
    ):  # used to be called "power_signal" which is misleading (not power)
        """ return a time-arry of the sum of spectrogram over all frequencies or a range of frequencies"""
        if freq_range is None:
            return np.sum(self.spectrogram, 0)
        else:
            return np.sum(self.bandpass(freq_range[0], freq_range[1]).spectrogram, 0)

    def net_amplitude(
        self, signal_band, reject_bands=None
    ):  # used to be called "net_power_signal" which is misleading (not power)
        """make amplitude signal of file f in signal_band and subtract amplitude from reject_bands

        rescale the signal and reject bands by dividing by their bandwidths in Hz 
        (amplitude of each reject_band is divided by the total bandwidth of all reject_bands.
        amplitude of signal_band is divided by badwidth of signal_band. ) 
        
        return: 1-d time series of net amplitude """

        # find the amplitude signal for the desired frequency band
        signal_band_amplitude = self.amplitude(signal_band)

        signal_band_bandwidth = signal_band[1] - signal_band[0]

        # rescale amplitude by 1 / size of frequency band in Hz ("amplitude per unit Hz" ~= color on a spectrogram)
        net_amplitude = signal_band_amplitude / signal_band_bandwidth

        # then subtract the energy in the the reject_bands from the signal_band_amplitude to get net_amplitude
        if not (reject_bands is None):
            # we sum up the sizes of the rejection bands (to not overweight signal_band)
            reject_bands = np.array(reject_bands)
            reject_bands_total_bandwidth = sum(reject_bands[:, 1] - reject_bands[:, 0])

            # subtract reject_band_amplitude
            for reject_band in reject_bands:
                reject_band_amplitude = self.amplitude(reject_band)
                net_amplitude = net_amplitude - (
                    reject_band_amplitude / reject_bands_total_bandwidth
                )

            # negative signal shouldn't be kept, because it means reject was stronger than signal. Zero it:
            net_amplitude = [max(0, s) for s in net_amplitude]

        return net_amplitude

    #     def save(self,destination):
    #         with open(destination,'wb') as file:
    #             pickle.dump(self,file)

    def to_image(self, shape=None, mode="RGB", spec_range=[-100, -20]):
        """
        create a Pillow Image from spectrogram
        linearly rescales values from db_range (default [-100, -20]) to [255,0]
        (ie, -20 db is loudest -> black, -100 db is quietest -> white)
        
        destination: a file path (string)
        shape=None: tuple of image dimensions, eg (224,224)
        mode="RGB": RGB for 3-channel color or "L" for 1-channel grayscale
        spec_range=[-100,-20]: the lowest and highest possible values in the spectrogram
            
        returns: Pillow Image
        """
        from PIL import Image

        # rescale values from db_range to [0,255]
        shift = spec_range[0]
        scale = 255.0 / (spec_range[1] - spec_range[0])
        array = (
            255.0 - (self.spectrogram - shift) * scale
        )  # reverse scale so that 0 is white

        # create and save pillow Image
        # we pass the array upside-down to create right-side-up image
        image = Image.fromarray(array[::-1, :])
        image = image.convert(mode)
        if shape is not None:
            image = image.resize(shape)

        return image
