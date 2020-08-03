#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np
from opensoundscape.audio import Audio
from opensoundscape.helpers import min_max_scale, linear_scale
import warnings
import pickle


class Spectrogram:
    """ Immutable spectrogram container
    """

    __slots__ = ("frequencies", "times", "spectrogram", "decibel_limits")

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
        super(Spectrogram, self).__setattr__("decibel_limits", (-100, -20))

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

        Args:
            window_type="hann": see scipy.signal.spectrogram docs for description of window parameter
            window_samples=512: number of audio samples per spectrogram window (pixel)
            overlap_samples=256: number of samples shared by consecutive windows
            decibel_limits = (-100,-20) : limit the dB values to (min,max) (lower values set to min, higher values set to max)

        Returns:
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
        # -> avoid RuntimeWarning by setting negative values to -np.inf (mapped to min_db later)
        spectrogram = 10 * np.log10(
            spectrogram, where=spectrogram > 0, out=np.full(spectrogram.shape, -np.inf)
        )

        # limit the decibel range (-100 to -20 dB by default)
        # values below lower limit set to lower limit, values above upper limit set to uper limit
        min_db, max_db = decibel_limits
        spectrogram[spectrogram > max_db] = max_db
        spectrogram[spectrogram < min_db] = min_db

        new_obj = cls(spectrogram, frequencies, times)
        super(Spectrogram, new_obj).__setattr__("decibel_limits", decibel_limits)
        return new_obj

    @classmethod
    def from_file(file,):
        """
        create a Spectrogram object from a file

        Args:
            file: path of image to load

        Returns:
            opensoundscape.spectrogram.Spectrogram object
        """

        raise NotImplementedError(
            "Loading Spectrograms from images is not implemented yet"
        )

    def __setattr__(self, name, value):
        raise AttributeError("Spectrogram's cannot be modified")

    def __repr__(self):
        return f"<Spectrogram(spectrogram={self.spectrogram.shape}, frequencies={self.frequencies.shape}, times={self.times.shape})>"

    def min_max_scale(self, feature_range=(0, 1)):
        """

        Linearly rescale spectrogram values to a range of values using
        in_range as minimum and maximum

        Args:
            feature_range: tuple of (low,high) values for output

        Returns:
            Spectrogram object with values rescaled to feature_range
        """

        if len(feature_range) != 2:
            raise AttributeError(
                "Error: `feature_range` doesn't look like a 2-element tuple?"
            )
        if feature_range[1] < feature_range[0]:
            raise AttributeError("Error: `feature_range` isn't increasing?")

        return Spectrogram(
            min_max_scale(self.spectrogram, feature_range=feature_range),
            self.frequencies,
            self.times,
        )

    def linear_scale(self, feature_range=(0, 1)):
        """

        Linearly rescale spectrogram values to a range of values
        using in_range as decibel_limits

        Args:
            feature_range: tuple of (low,high) values for output

        Returns:
            Spectrogram object with values rescaled to feature_range
        """

        if len(feature_range) != 2:
            raise AttributeError(
                "Error: `feature_range` doesn't look like a 2-element tuple?"
            )
        if feature_range[1] < feature_range[0]:
            raise AttributeError("Error: `feature_range` isn't increasing?")

        return Spectrogram(
            linear_scale(
                self.spectrogram, in_range=self.decibel_limits, out_range=feature_range
            ),
            self.frequencies,
            self.times,
        )

    def limit_db_range(self, min_db=-100, max_db=-20):
        """ Limit the decibel values of the spectrogram to range from min_db to max_db

            values less than min_db are set to min_db
            values greater than max_db are set to max_db

            similar to Audacity's gain and range parameters

            Args:
                min_db: values lower than this are set to this
                max_db: values higher than this are set to this
            Returns:
                Spectrogram object with db range applied
        """
        _spec = self.spectrogram

        _spec[_spec > max_db] = max_db
        _spec[_spec < min_db] = min_db

        return Spectrogram(_spec, self.frequencies, self.times)

    def bandpass(self, min_f, max_f):
        """ extract a frequency band from a spectrogram

        crops the 2-d array of the spectrograms to the desired frequency range

        Args:
            min_f: low frequency in Hz for bandpass
            high_f: high frequency in Hz for bandpass

        Returns:
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

        Args:
            start_time: in seconds
            end_time: in seconds

        Returns:
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
        """Plot the spectrogram with matplotlib.pyplot

        Args:
            inline=True:
            fname=None: specify a string path to save the plot to (ending in .png/.pdf)
            show_colorbar: include image legend colorbar from pyplot
        """
        from matplotlib import pyplot as plt

        plt.pcolormesh(self.times, self.frequencies, self.spectrogram, shading="auto")
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

    def amplitude(self, freq_range=None):
        """create an amplitude vs time signal from spectrogram

        by summing pixels in the vertical dimension

        Args
            freq_range=None: sum Spectrogrm only in this range of [low, high] frequencies in Hz
            (if None, all frequencies are summed)

        Returns:
            a time-series array of the vertical sum of spectrogram value

        """
        if freq_range is None:
            return np.sum(self.spectrogram, 0)
        else:
            return np.sum(self.bandpass(freq_range[0], freq_range[1]).spectrogram, 0)

    def net_amplitude(
        self, signal_band, reject_bands=None
    ):  # used to be called "net_power_signal" which is misleading (not power)
        """create amplitude signal in signal_band and subtract amplitude from reject_bands

        rescale the signal and reject bands by dividing by their bandwidths in Hz
        (amplitude of each reject_band is divided by the total bandwidth of all reject_bands.
        amplitude of signal_band is divided by badwidth of signal_band. )

        Args:
            signal_band: [low,high] frequency range in Hz (positive contribution)
            reject band: list of [low,high] frequency ranges in Hz (negative contribution)

        return: time-series array of net amplitude """

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

        Args:
            destination: a file path (string)
            shape=None: tuple of image dimensions, eg (224,224)
            mode="RGB": RGB for 3-channel color or "L" for 1-channel grayscale
            spec_range=[-100,-20]: the lowest and highest possible values in the spectrogram

        Returns:
            Pillow Image object
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
