#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np
from opensoundscape.audio import Audio
import warnings

class Spectrogram:
    """ Immutable spectrogram container
    """

    __slots__ = ("frequencies", "times", "spectrogram")

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
    def from_audio(cls, audio, window="hann", overlap=75, segment_length=512, decibels=True):
        if not isinstance(audio, Audio):
            raise TypeError("Class method expects Samples class as input")

        frequencies, times, spectrogram = signal.spectrogram(
            audio.samples,
            audio.sample_rate,
            window=window,
            nperseg=segment_length,
            noverlap=segment_length * overlap / 100,
            scaling="spectrum",
        )
        
        if decibels:
            spectrogram = 10 * np.log10(spectrogram)

        return cls(spectrogram, frequencies, times)

    def __setattr__(self, name, value):
        raise AttributeError("Spectrogram's cannot be modified")

    def __repr__(self):
        return f"<Spectrogram(spectrogram={self.spectrogram.shape}, frequencies={self.frequencies.shape}, times={self.times.shape})>"

#     def decibel_filter(self, decibel_threshold=-100.0):
#         """ Apply a decibel based floor function
                #spectrograms are now in decibels by default
#         """

#         remove_zeros = np.copy(self.spectrogram)
#         remove_zeros[remove_zeros == 0.0] = np.nan
#         in_decibel = 10.0 * np.log10(remove_zeros)
#         in_decibel[in_decibel <= decibel_threshold] = decibel_threshold
#         return Spectrogram(
#             np.nan_to_num(10.0 ** (in_decibel / 10.0)), self.frequencies, self.times
#         )

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

    def audacity_gain_range(self, spec_gain=20, spec_range=80):
        """ Apply gain and range similar to Audacity
        """

        _spec = self.spectrogram
        _spec[_spec > -spec_gain] = -spec_gain
        _spec[_spec < -(spec_gain + spec_range)] = -(spec_gain + spec_range)
        return Spectrogram(_spec, self.frequencies, self.times)

    def bandpass(self, freq_range):
        """ extract a frequency band from a spectrogram

        params:
        freq_range: [low, high] frequencies in Hz for bandpass

        returns:
        bandpassed spectrogram 

        """

        min_f = freq_range[0]
        max_f = freq_range[1]

        #find indices of the frequencies in spec_freq closest to min_f and max_f
        lowest_index = np.abs(self.frequencies - min_f).argmin() 
        highest_index = np.abs(self.frequencies - max_f).argmin()

        #take slices of the spectrogram and spec_freq that fall within desired range
        return Spectrogram(
            self.spectrogram[lowest_index:highest_index, :],
            self.frequencies[lowest_index:highest_index],
            self.times
        )
    
    def trim(self, time_range):
        """ extract a time segment from a spectrogram

        params:
        freq_range: [start_time, end_time] in seconds

        returns:
        trimmed spectrogram 
        """

        min_t = time_range[0]
        max_t = time_range[1]

        #find indices of the times in self.times closest to min_t and max_t
        lowest_index = np.abs(self.times - min_t).argmin() 
        highest_index = np.abs(self.times - max_t).argmin()

        #take slices of the spectrogram and spec_freq that fall within desired range
        return Spectrogram(
            self.spectrogram[:, lowest_index:highest_index],
            self.frequencies,
            self.times[lowest_index:highest_index]
        )
    
    def plot(self,inline=True,fname=None,show_colorbar=False):
        """Plot the spectrogram with pyplot. 
        
        Parameters:
            inline=True: 
            fname=None: specify a string path to save the plot to (ending in .png/.pdf)
            show_colorbar: include image legend colorbar from pyplot
        """
        from matplotlib import pyplot as plt
        
        plt.pcolormesh(self.times,self.frequencies,self.spectrogram)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        if show_colorbar:
            plt.colorbar()

        #if fname is not None, save to file path fname
        if fname:
            plt.savefig(fname)
        
        #if not saving to file, check if a matplotlib backend is available 
        if inline:
            import os
            if os.environ.get('MPLBACKEND') is None:
                warnings.warn("MPLBACKEND is 'None' in os.environ. Skipping plot.")
            else:
                plt.show()       

    
    def power_signal(self, freq_range=None):
        """ return a time-arry of the sum of spectrogram over all frequencies or a range of frequencies"""
        if freq_range is None:
            return np.sum(self.spectrogram,0)
        else:
            return np.sum(self.bandpass(freq_range).spectrogram,0)
    
    def net_power_signal(self, signal_band, reject_bands = None):
        """make power signal of file f in signal_band and subtract power from reject_bands

        rescale the signal and reject bands by dividing by their bandwidths in Hz 
        (power of each reject_band is divided by the total bandwidth of all reject_bands.
        power of signal_band is divided by badwidth of signal_band. ) 
        
        return: 1-d time series of net power """

        #find the power signal for the desired frequency band
        signal_band_power = self.power_signal(signal_band)

        signal_band_bandwidth = signal_band[1]-signal_band[0]

        #rescale power by 1 / size of frequency band in Hz ("power per unit Hz" ~= color on a spectrogram)
        net_power = signal_band_power / signal_band_bandwidth 

        #then subtract the energy in the the reject_bands from the signal_band_power to get net_power
        if not (reject_bands is None): 
            #we sum up the sizes of the rejection bands (to not overweight signal_band)
            reject_bands = np.array(reject_bands)
            reject_bands_total_bandwidth = sum(reject_bands[:,1]-reject_bands[:,0])

            #subtract reject_band_power 
            for reject_band in reject_bands: 
                reject_band_power = self.power_signal(reject_band)
                net_power = net_power - (reject_band_power/reject_bands_total_bandwidth)

            #negative signal shouldn't be kept, because it means reject was stronger than signal. Zero it:
            net_power = [max(0,s) for s in net_power]

        return net_power
