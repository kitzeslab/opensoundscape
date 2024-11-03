import warnings
import numpy as np
import datetime
import pandas as pd

from opensoundscape.audio import Audio
from opensoundscape import audio
from opensoundscape.utils import cast_np_to_native
from opensoundscape.localization.localization_algorithms import SPEED_OF_SOUND
from opensoundscape.localization import localization_algorithms
from opensoundscape.localization.position_estimate import PositionEstimate


class SpatialEvent:
    """
    Class that estimates the location of a single sound event

    Uses receiver locations and time-of-arrival of sounds to estimate
    sound source location
    """

    def __init__(
        self,
        receiver_files,
        receiver_locations,
        max_delay,
        min_n_receivers=3,
        receiver_start_time_offsets=None,
        start_timestamp=None,
        duration=None,
        class_name=None,
        bandpass_range=None,
        cc_threshold=0,
        cc_filter=None,
        speed_of_sound=SPEED_OF_SOUND,
    ):
        """
        Initialize SpatialEvent

        Args:
            receiver_files: list of audio files, one for each receiver
            receiver_locations: list of [x,y] or [x,y,z] positions of each receiver in meters
            max_delay: maximum time delay (in seconds) to consider for time-delay-of-arrival estimate. Cannot be longer than 1/2 the duration.
            receiver_start_time_offsets: list of start_time of detection (seconds) for each receiver relative to start of audio file
                - if all audio files started at the same real-world time, this value will be the same for all recievers
                - for example, 5.0 means the detection window starts 5 seconds after the beginning of the Audio file
                (the detection window's duration in seconds is given by the `duration` argument and is the same across recievers)
                - if `None`, and `start_timestamp` is given, attempts to extract the correct audio time period from each file
                    using the audio file's metadata 'recording_start_time' field (using Audio.from_file with start_timestamp argument)
            start_timestamp: start time of detection as datetime.datetime
            duration: duration of sound event. This duration of audio will be used for cross-correlation to estimate TDOAs.
            class_name: (str) name of detection's class. Default: None
            bandpass_range: [low,high] frequency range that audio will be bandpassed to before cross-correlation.
                default: None. No bandpassing. Bandpassing audio to the frequency range of the relevant sound is recommended for best cross correlation results.
            cc_threshold: float. This acts as a minimum threshold for cross correlation. If the cross correlation at the estimated time delay is is below this value, the corresponding time delay is discarded and not used during localization.
                NOTE: The scale of the cross correlation values depends on the cc_filter used.
                default: None. Do not discard any time delays.
            speed_of_sound: float, optional. Speed of sound in meters per second.
                Default: opensoundscape.localization.localization_algorithms.SPEED_OF_SOUND

        Methods:
            estimate_location:
                - Estimates the tdoas using cross_correlation if not already estimated.
                - Estimates the location of the event using the tdoas and receiver locations.
                - returns a PositionEstimate object with .location_estimate and other attributes
                    (if localization is not successful, .location_estimate is None)

        Editable Attributes:
            These are parameters that can be set before calling estimate_location()
            min_n_receivers: minimum number of receivers that must detect an event for it to be localized
            cc_threshold: threshold for cross correlation
            cc_filter: filter for generalized cross correlation, see
                opensoundscape.signal_processing.gcc()
            max_delay: only delays in +/- this range (seconds) will be considered for possible delay
                (see opensoundscape.signal_processing.tdoa());
            bandpass_range: bandpass audio to [low, high] frequencies in Hz before cross correlation
            speed_of_sound: speed of sound in meters per second

        Static Attributes:
            receiver_files: list of audio files, one for each receiver
            receiver_locations: list of [x,y] or [x,y,z] positions of each receiver in meters
            start_timestamp: start time of detection as datetime.datetime
            duration: length in seconds of the event
            class_name: name of detection's class

        Computed Attributes:
            tdoas: time delay at each receiver (computed by _estimate_delays())
            cc_maxs: max of cross correlation for each time delay (computed by _estimate_delays())
        """
        # editable attributes
        self.min_n_receivers = min_n_receivers
        self.cc_threshold = cc_threshold
        self.cc_filter = cc_filter
        self.max_delay = max_delay
        self.bandpass_range = bandpass_range
        self.speed_of_sound = speed_of_sound

        # static attributes
        self.receiver_files = np.array(receiver_files)
        self.receiver_locations = np.array(receiver_locations)
        self.start_timestamp = start_timestamp
        self.duration = duration
        self.class_name = class_name
        if receiver_start_time_offsets is not None:  # cast to np array
            self.receiver_start_time_offsets = np.array(receiver_start_time_offsets)
        else:  # not provided; keep None value
            self.receiver_start_time_offsets = None

        # Verify that max_delay is not longer than the duration of the audio and raise a value error if it is
        if self.max_delay >= self.duration:
            raise ValueError(
                f"max_delay ({self.max_delay}) is longer than duration ({self.duration}) of audio clips."
            )

        # computed attributes
        self.tdoas = None  # time delay at each receiver
        self.cc_maxs = None  # max of cross correlation for each time delay

    def estimate_location(
        self,
        localization_algorithm="gillette",
        use_stored_tdoas=True,
    ):
        """
        Estimate spatial location of this event.

        This method first estimates the time delays (TDOAS) using cross-correlation, then estimates
        the location from those TDOAS. Localization is performed in 2d or 3d according to the
        dimensions of self.receiver_locations (x,y) or (x,y,z) Note: if self.tdoas or
        self.receiver_locations is None, first calls self._estimate_delays() to estimate the time
        delays.

        If you want to change some parameters of the localization (e.g. try a different
        cc_threshold, or bandpass_range), you can set the appropriate attribute (e.g.
        self.cc_threshold = 0.01) before calling self.estimate_location().

        Args:
            - localization_algorithm: algorithm to use for estimating the location of a sound event
              from the locations and time delays of a set of detections. Options are 'gillette' or
              'soundfinder'. Default is 'gillette'.
            - use_stored_tdoas: if True, uses the tdoas stored in self.tdoas to estimate the
              location.
                If False, first calls self._estimate_delays() to estimate the tdoas. default: True

        Returns:
            PositionEstimate object with .location_estimate and other attributes
            - if localization is not successful, .location_estimate attribute of returned object is
              None
        """
        # If no values are already stored, perform generalized cross correlation to estimate time delays
        # or if user wants to re-estimate the time delays, perform generalized cross correlation to estimate time delays
        if self.tdoas is None or self.cc_maxs is None or use_stored_tdoas is False:
            # Stores the results in the the attributes self.cc_maxs and self.tdoas
            # if it fails, they will be None
            self._estimate_delays()

        # if self.tdoas is still not set, don't attempt to localize (_estimate_delays was not successful)
        if self.tdoas is None:
            return PositionEstimate(location_estimate=None)
        else:
            # creates a PositionEstimate object with .location_setimate and other attributes
            return self._localize_after_cross_correlation(
                localization_algorithm=localization_algorithm
            )

    def _calculate_receiver_start_time_offsets(self):
        """attempt to calculate starting position (second) from audio file to this event

        uses self.start_timestamp and self.receiver_files, which should point to audio files
        with localized start timestamps in the metadata field `recording_start_time`
        (when parsed with audio.parse_metadata(path))

        Effects:
            sets self.receiver_start_time_offsets with the calculated values
        """
        event_start_timestamp = self.start_timestamp
        if isinstance(event_start_timestamp, pd.Timestamp):
            event_start_timestamp = event_start_timestamp.to_pydatetime()

        # get the start timestamp for each audio file
        file_start_timestamp = [
            audio.parse_metadata(file)["recording_start_time"]
            for file in self.receiver_files
        ]

        # then, calculate the offset from the event start time to each file's start time
        self.receiver_start_time_offsets = [
            (event_start_timestamp - receiver_start_timestamp).total_seconds()
            for receiver_start_timestamp in file_start_timestamp
        ]

    def _estimate_delays(self):
        """Hidden method to estimate time delay of event relative to receiver_files[0] with gcc

        Performs Generalized Cross Correlation of each file against the first,
            extracting the segment of audio of length self.duration at self.start_time

        Assumes audio files are synchronized such that they start at the same time

        Uses the following attributes of the object:
            cc_filter: filter for generalized cross correlation, see
                opensoundscape.signal_processing.gcc()
            max_delay: only delays in +/- this range (seconds) will be considered for possible delay
                (see opensoundscape.signal_processing.tdoa());
                If None, defaults to self.max_delay
            bandpass_range: bandpass audio to [low, high] frequencies in Hz before
                cross correlation
                If None, defaults to self.bandpass_range=

        Returns:
            list of time delays, list of max cross correlation values

            each list is the same length as self.receiver_files, and each
            value corresponds to the cross correlation of one file relative
            to the first file (self.receiver_files[0])

        Effects:
            sets self.tdoas and self.cc_maxs with the same values as those returned
        """

        extracted_clip_duration = self.duration + 2 * self.max_delay

        if self.receiver_start_time_offsets is None:
            assert (
                self.start_timestamp is not None
            ), "must set .receiver_start_time_offsets or .start_timestamp. Both were None."
            # sets the .receiver_start_time_offsets using start_timestamp and receiver start times
            self._calculate_receiver_start_time_offsets()

        # load audio from desired time period
        reference_audio = Audio.from_file(
            self.receiver_files[0],
            offset=self.receiver_start_time_offsets[0] - self.max_delay,
            duration=extracted_clip_duration,
        )

        # make sure the audio clip is of the desired length
        # Note: this will give errors for clips starting at 0s since we can't get earlier audio;
        # alternatively, we could pad the audio with zeros
        if not np.isclose(reference_audio.duration, extracted_clip_duration, atol=1e-3):
            # don't try to localize
            self.error_msg = "did not get audio clip of desired length"
            self.tdoas = None
            self.cc_maxs = None
            return None, None

        # bandpass once now to avoid repeating operation for each receiver
        if self.bandpass_range is not None:
            reference_audio = reference_audio.bandpass(
                low_f=self.bandpass_range[0], high_f=self.bandpass_range[1], order=9
            )

        # estimate time difference of arrival (tdoa) for each file relative to the first
        # skip the first because we don't need to cross correlate a file with itself
        tdoas = []
        cc_maxs = []

        # catch the receivers that have an issue and should be discarded
        # e.g. their file starts or end during the time-window, so estimate_delays is not possible
        bad_receivers_index = []

        for index, file in enumerate(self.receiver_files):
            if index == 0:  # can skip reference audio cc with itself
                tdoas.append(0)  # first file's delay to itself is zero
                cc_maxs.append(1)  # set first file's cc_max to 1
                continue

            # use specified time offsets to extract the correct audio segment
            audio2 = Audio.from_file(
                file,
                offset=self.receiver_start_time_offsets[index] - self.max_delay,
                duration=extracted_clip_duration,
            )

            # catch edge cases where the audio lengths do not match.
            if (
                abs(len(audio2.samples) - len(reference_audio.samples)) > 1
            ):  # allow for 1 sample difference
                bad_receivers_index.append(index)
            else:
                tdoa, cc_max = audio.estimate_delay(
                    primary_audio=audio2,
                    reference_audio=reference_audio,
                    max_delay=self.max_delay,
                    bandpass_range=self.bandpass_range,
                    cc_filter=self.cc_filter,
                    return_cc_max=True,
                    skip_ref_bandpass=True,
                )
                tdoas.append(tdoa)
                cc_maxs.append(cc_max)

        self.tdoas = np.array(tdoas)
        self.cc_maxs = np.array(cc_maxs)

        # delete the bad receivers from this SpatialEvent
        if len(bad_receivers_index) > 0:
            print(
                f"Warning: {len(bad_receivers_index)} receivers were discarded because their audio files were not the same length as the primary receiver."
            )
            # drop the bad receivers from the list of files and locations
            self.receiver_files = [
                file
                for index, file in enumerate(self.receiver_files)
                if index not in bad_receivers_index
            ]

            self.receiver_locations = np.array(
                [
                    location
                    for index, location in enumerate(self.receiver_locations)
                    if index not in bad_receivers_index
                ]
            )

        return self.tdoas, self.cc_maxs

    def _localize_after_cross_correlation(self, localization_algorithm):
        """
        Hidden method to estimate location from time delays and receiver locations
        """

        # filter by cross correlation threshold, removing time delays + locations
        # if cross correlation did not exceed a minimum value
        # (low max cross correlation values indicate low confidence that the time
        # delay truly represents two recordings of the same sound event)
        tdoas = self.tdoas
        locations = self.receiver_locations

        # apply the cc_threshold filter
        # only keep receivers that have a cc_max above the cc_threshold
        rec_mask = self.cc_maxs > self.cc_threshold
        tdoas = tdoas[rec_mask]
        locations = locations[rec_mask]

        # If there aren't enough receivers, don't attempt localization.
        if len(tdoas) < self.min_n_receivers:
            return PositionEstimate(location_estimate=None)

        # Estimate location from receiver locations and time differences of arrival
        location_estimate = localization_algorithms.localize(
            receiver_locations=locations,
            tdoas=tdoas,
            algorithm=localization_algorithm,
            speed_of_sound=self.speed_of_sound,
        )

        # Store the distance residuals (only for the receivers used) in PositionEstimate
        distance_residuals = None
        if location_estimate is not None:
            distance_residuals = calculate_tdoa_residuals(
                receiver_locations=locations,
                tdoas=tdoas,
                location_estimate=location_estimate,
                speed_of_sound=self.speed_of_sound,
            )

        return PositionEstimate(
            location_estimate=location_estimate,
            class_name=self.class_name,
            receiver_files=self.receiver_files[rec_mask],
            receiver_locations=locations,
            tdoas=tdoas,
            cc_maxs=self.cc_maxs[rec_mask],
            start_timestamp=self.start_timestamp,
            receiver_start_time_offsets=self.receiver_start_time_offsets[rec_mask],
            duration=self.duration,
            distance_residuals=distance_residuals,
        )

    @classmethod
    def from_dict(cls, dictionary):
        """Recover SpatialEvent from dictionary, eg loaded from json"""
        array_keys = (
            "receiver_locations",
            "tdoas",
            "cc_maxs",
            "location_estimate",
            "distance_residuals",
            "receivers_used_for_localization",
        )
        s = cls(
            receiver_files=dictionary["receiver_files"],
            receiver_locations=dictionary["receiver_locations"],
            max_delay=dictionary["max_delay"],
            duration=dictionary["duration"],
        )
        for key in dictionary.keys():
            val = dictionary[key]
            if key == "start_timestamp":
                val = pd.Timestamp(val)
            if key in array_keys:
                val = np.array(val)
            s.__setattr__(key, val)
        return s

    def to_dict(self):
        """SpatialEvent to json-able dictionary"""
        d = self.__dict__.copy()
        for key in d.keys():
            if isinstance(d[key], np.ndarray):
                d[key] = d[key].tolist()
            if isinstance(d[key], pd.Timestamp):
                d[key] = str(d[key])
        return d


def events_to_df(list_of_events):
    """convert a list of SpatialEvent objects to pd DataFrame

    Args:
        list_of_events: list of SpatialEvent objects

    Returns:
        pd.DataFrame with columns for each attribute of a SpatialEvent object

    See also: df_to_events, SpatialEvent.from_dict, SpatialEvent.to_dict
    """
    return pd.DataFrame([e.__dict__ for e in list_of_events])


def df_to_events(df):
    """convert a pd DataFrame to list of SpatialEvent objects

    works best if df was created using `events_to_df`

    Args:
        df: pd.DataFrame with columns for each attribute of a SpatialEvent object

    Returns:
        list of SpatialEvent objects

    See also: events_to_df, SpatialEvent.from_dict, SpatialEvent.to_dict
    """
    return [SpatialEvent.from_dict(df.loc[i].to_dict()) for i in df.index]


def calculate_tdoa_residuals(
    receiver_locations, tdoas, location_estimate, speed_of_sound
):
    """
    Calculate the residual distances of the TDOA localization algorithm

    The residual represents the discrepancy between (difference in distance
    of each reciever to estimated location) and (observed tdoa), and has
    units of meters. Residuals are calculated as follows:

        expected = calculated time difference of arrival between reference and
            another receiver, based on the locations of the receivers and
            estimated event location
        observed = observed tdoas provided to localization algorithm

        residual time = expected - observed (in seconds)

        residual distance = speed of sound * residual time (in meters)

    Args:
        receiver_location: The list of coordinates (in m) of each receiver,
            as [x,y] for 2d or or [x,y,z] for 3d.
        tdoas: List of time delays of arival for the sound at each receiver,
            relative to the first receiver in the list (tdoas[0] should be 0)
        location_estimate: The estimated location of the sound, as (x,y) or (x,y,z) in meters
        speed_of_sound: The speed of sound in m/s

    Returns:
        np.array containing the residuals in units of meters, one per receiver
    """
    # ensure all are numpy arrays
    receiver_locations = np.array(receiver_locations)
    tdoas = np.array(tdoas)
    location_estimate = np.array(location_estimate)

    # Calculate the TDOA residuals

    # calculate time sound would take to travel from the estimated location
    # to each receiver (distance/speed=time)
    distances = [np.linalg.norm(r - location_estimate) for r in receiver_locations]
    travel_times = np.array(distances) / speed_of_sound

    # the expected time _difference_ of arrival for any receiver vs the
    # reference receiver is the difference in travel times from the
    # location estimate to each of the receivers compared to the first
    expected_tdoas = travel_times - travel_times[0]

    # the time residual is the difference between the observed tdoa values
    # and those expected according to the estimated location
    # first value will be 0 by definition
    time_residuals = expected_tdoas - tdoas

    # convert residuals from units of time (s) to distance (m) via speed of sound
    return time_residuals * speed_of_sound


def localize_events_parallel(events, num_workers, localization_algorithm):

    # perform gcc to estimate relative time of arrival at each receiver
    # estimate locations of sound event using time delays and receiver locations
    # this calls estimate_delays under the hood
    from joblib import Parallel, delayed

    # parallelize the localization of each event across cpus
    # return list of PositionEstimate objects
    return Parallel(n_jobs=num_workers)(
        delayed(e.estimate_location)(localization_algorithm=localization_algorithm)
        for e in events
    )
