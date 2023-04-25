"""Tools for localizing audio events from synchronized recording arrays"""
import warnings
import numpy as np
import pandas as pd
from opensoundscape.audio import Audio
from scipy.signal import correlate, correlation_lags

import opensoundscape.signal_processing as sp
from opensoundscape import audio

# define defaults for physical constants
SPEED_OF_SOUND = 343  # default value in meters per second


class InsufficientReceiversError(Exception):
    """raised when there are not enough receivers to localize an event"""

    pass


class SpatialEvent:
    """
    Class that estimates the position of a single sound event

    Uses reciever positions and time-of-arrival of sounds to estimate
    soud source position
    """

    def __init__(
        self,
        receiver_files,
        receiver_positions,
        start_time=0,
        duration=None,
        class_name=None,
        bandpass_range=None,
        cc_threshold=None,
        max_delay=None,
    ):
        """initialize SpatialEvent

        Args:
            receiver_files: list of audio files, one for each reciever
            receiver_positions: list of [x,y] or [x,y,z] cartesian position of each receiver in meters
            start_time: start position of detection relative to start of audio file, for cross correlation
            duration: duration of audio segment to use for cross-correlation
            class_name=None: (str) name of detection's class
            tdoas=None: optionally specify relative time difference of arrival of event at each receiver, in
                seconds. If not None, should be list of same length as receiver_files and receiver_positions
            bandpass_range: [low,high] frequency for audio bandpass before cross-correlation
                default [None]: does not perform audio bandpass before cross-correlation
            cc_threshold: float, default=None. During localization from time delays, discard time delays and
                associated positions if max cross correlation value was lower than this threshold.
                default: None uses all delays and positions regardless of max cc value
            max_delay: maximum time delay (in seconds) to consider for cross correlation
                (see `opensoundscape.signal_processing.tdoa`)

        Methods:
            estimate_delays:
                use generalized cross correlation to find time delays of arrival
                of the event at each receiver
            estimate_location:
                perform tdoa based position estimation
                - calls estimate_delays() if needed
            calculate_distance_residuals:
                compute residuals (descrepancies) between tdoa and estimated position
            calculate_residual_rmse:
                compute the root mean square value of the tdoa distance residuals

        """
        self.receiver_files = receiver_files
        self.receiver_positions = receiver_positions
        self.start_time = start_time
        self.duration = duration
        self.class_name = class_name
        self.bandpass_range = bandpass_range
        self.cc_threshold = cc_threshold
        self.max_delay = max_delay

        # initialize attributes to store values calculated by methods
        self.tdoas = None  # time delay at each receiver
        self.cc_maxs = None  # max of cross correlation for each time delay
        self.position_estimate = None  # cartesian position estimate in meters

        # could implement this later:
        # hidden attributes store estimates and error metrics
        # from gillette and soundfinder localization algorithms
        # self._gillette_position_estimate = None
        # self._gillette_error = None
        # self._soundfinder_position_estimate = None
        # self._soundfinder_pseudorange_error = None

    def estimate_location(
        self,
        algorithm="gillette",
        cc_threshold=None,
        min_num_receivers=3,
        speed_of_sound=SPEED_OF_SOUND,
    ):
        """
        estimate spatial location of this event

        uses self.tdoas and self.receiver_positions to estimate event location

        Note: if self.tdoas or self.receiver_positions is None, first calls
        self.estimate_delays() to estimate these values

        Localization is performed in 2d or 3d according to the dimensions of
        self.receiver_positions (x,y) or (x,y,z)

        Args:
            algorithm: 'gillette' or 'soundfinder', see localization.localize()
            cc_threshold: see SpatialEvent documentation
            min_num_receivers: if number of receivers with cross correlation exceeding
                the threshold is fewer than this, raises InsufficientReceiversError
                instead of estimating a spatial position

        Returns:
            position estimate as cartesian coordinates (x,y) or (x,y,z) (units: meters)

        Raises:
            InsufficientReceiversError if the number of receivers with cross correlation
                maximums exceeding `cc_threshold` is less than `min_num_receivers`

        Effects:
            sets the value of self.position_estimate to the same value as the returned position
        """
        if cc_threshold is None:
            cc_threshold = self.cc_threshold

        # perform generalized cross correlation to estimate time delays
        # (unless values are already stored in attributes)
        if self.tdoas is None or self.cc_maxs is None:
            self.estimate_delays()

        # filter by cross correlation threshold, removing time delays + positions
        # if cross correlation did not exceed a minimum value
        # (low max cross correlation values indicate low confidence that the time
        # delay truly represents two recordings of the same sound event)
        tdoas = self.tdoas
        positions = self.receiver_positions
        if cc_threshold is not None:
            tdoas = tdoas[self.cc_maxs > cc_threshold]
            positions = positions[self.cc_maxs > cc_threshold]

        # assert there are enough receivers remaining to localize the event
        if len(tdoas) < min_num_receivers:
            raise InsufficientReceiversError(
                f"Number of tdoas exceeding cc threshold ({len(tdoas)} was fewer "
                f"than min_num_receivers ({min_num_receivers})"
            )

        # estimate location from receiver positions and relative time of arrival
        # TODO: enable returning error estimates
        self.position_estimate = localize(
            receiver_positions=positions,
            tdoas=tdoas,
            algorithm=algorithm,
            speed_of_sound=speed_of_sound,
        )

        return self.position_estimate

    def estimate_delays(self, bandpass_range=None, cc_filter="phat", max_delay=None):
        """estimate time delay of event relative to receiver_files[0] with gcc

        Performs Generalized Cross Correlation of each file against the first

        Assumes audio files are synchronized such that they start at the same time

        Args:
            bandpass_range: bandpass audio to [low, high] frequencies in Hz before
                cross correlation; if None, defaults to self.bandpass_range
            cc_filter: filter for generalized cross correlation, see
                opensoundscape.signal_processing.gcc()
            max_delay: see opensoundscape.signal_processing.tdoa()

        Returns:
            list of time delays, list of max cross correlation values

            each list is the same length as self.receiver_files, and each
            value corresponds to the cross correlation of one file relative
            to the first file (self.receiver_files[0])

        Effects:
            sets self.tdoas and self.cc_maxs with the same values returned
        """

        # user can provide audio bandpass range, otherwise use saved attribute
        # bandpass range is a tuple of [low freq, high freq] in Hz
        if bandpass_range is None:
            bandpass_range = self.bandpass_range
        # same for max_delay
        if max_delay is None:
            max_delay = self.max_delay

        start, dur = self.start_time, self.duration
        audio1 = Audio.from_file(self.receiver_files[0], offset=start, duration=dur)

        # bandpass once now to avoid repeating operation for each receiver
        if bandpass_range is not None:
            audio1 = audio1.bandpass(bandpass_range[0], bandpass_range[1])

        # estimate time difference of arrival (tdoa) for each file relative to the first
        # skip the first because we don't need to cross correlate a file with itself
        tdoas = [0]  # first file's delay to itself is zero
        cc_maxs = [1]
        for file in self.receiver_files[1:]:
            audio2 = Audio.from_file(file, offset=start, duration=dur)
            tdoa, cc_max = audio.estimate_delay(
                audio=audio2,
                reference_audio=audio1,
                bandpass_range=bandpass_range,
                cc_filter=cc_filter,
                return_cc_max=True,
                max_delay=max_delay,
                skip_ref_bandpass=True,
            )
            tdoas.append(tdoa)
            cc_maxs.append(cc_max)

        self.tdoas = tdoas
        self.cc_maxs = cc_maxs

        return tdoas, cc_maxs

    def calculate_distance_residuals(self, speed_of_sound=SPEED_OF_SOUND):
        """calculate distance residuals for each receiver from tdoa localization

        The residual represents the discrepancy between (difference in distance
        of each reciever to estimated position) and (observed tdoa), and has
        units of meters.

        Args:
            speed_of_sound: speed of sound in m/s

        Returns:
            array of residuals (units are meters), one for each receiver

        Effects:
            stores the returned residuals as self.distance_residuals
        """
        if self.tdoas is None or self.position_estimate is None:
            warnings.warn(
                "missing self.tdoas or self.position_estimate, "
                "returning None for distance_residuals"
            )
            return None

        # store the calcualted tdoas as an attribute
        self.distance_residuals = calculate_tdoa_residuals(
            receiver_positions=self.receiver_positions,
            tdoas=self.tdoas,
            position_estimate=self.position_estimate,
            speed_of_sound=speed_of_sound,
        )

        # return the same residuals
        # TODO I think this is bad programming because the array could be modified
        # same issue with other methods of this class
        return self.distance_residuals

    def calculate_residual_rms(self, speed_of_sound=SPEED_OF_SOUND):
        """calculate the root mean square distance residual from tdoa localization

        Args:
            speed_of_sound: speed of sound in meters per second

        Returns:
            root mean square value of residuals, in meters
            - returns None if self.tdoas or self.position_estimate
                are None

        See also: `SpatialEvent.calculate_distance_residuals()`
        """
        if self.tdoas is None or self.position_estimate is None:
            warnings.warn(
                "missing `self.tdoas` or `self.position_estimate`, "
                "returning None for residual rms"
            )
            return None

        # calculate the residual distance for each reciever
        # this represents the discrepancy between (difference in distance
        # of each reciever to estimated position) and (observed tdoa)
        residuals = self.calculate_distance_residuals(speed_of_sound)
        return np.sqrt(np.mean(residuals**2))


class SynchronizedRecorderArray:
    """
    Class with utilities for localizing sound events from array of recorders

    Algorithm
    ----------
    The user provides a table of class detections from each recorder with timestamps. The user
    also provides a table listing the spatial location of the recorder for each unique audio
    file in the table of detections. The audio recordings must be synchronized
    such that timestamps from each recording correspond to the exact same real-world time.

    Localization of sound events proceeds in three steps:

    1. Grouping of detections into candidate events:

        Simultaneous and spatially clustered detections of a class are selected as targets
        for localization of a single real-world sound event.

        For each detection of a species, the grouping algorithm treats the reciever with the detection
        as a "reference receiver", then selects all detections of the species at the same time and
        within `max_distance_between_receivers` of the reference reciever (the "surrounding detections").
        This selected group of simulatneous, spatially-clustered detections of a class beomes one
        "candidate event" for subsequent localization.

        If the number of recorders in the candidate event is fewer than `min_number_of_receivers`, the
        candidate event is discarded.

        This step creates a highly redundant set of candidate events to localize, because each detection
        is treated separately with its recorder as the 'reference recorder'. Thus, the localized events created by this algorithm may contain multiple instances representing
        the same real-world sound event.


    2. Estimate time delays with cross correlation:

        For each candidate event, the time delay between the reference reciever's detection and the
        surrounding recorders' detections is estimated through generalized cross correlation.

        If the max value of the cross correlation is below `cc_threshold`, the corresponding time delay
        is discarded and not used during localization. This provides a way of filtering out
        undesired time delays that do not correspond to two recordings of the same sound event.

        If the number of time delays in the candidate event is fewer than `min_number_of_receivers`
        after filtering by cross correlation threshold, the candidate event is discarded.

    3. Estiamte positions

        The position of the event is estimated based on the positions and time delays of
        each detection.

        Position estimation from the positions and time delays at a set of receivers is performed
        using one of two algorithms, described in `localization_algorithm` below.

    4. Filter by residual error

        The residual errors represent descrepencies between (a) time of arrival of
        the event at a reciever and (b) distance from reciever to estimated position.

        Estimated positions are discarded if the root mean squared residual error is
        greater than `residual_rmse_threshold` #TODO implement?


    Parameters
    ----------
    files : list
        List of synchronized audio files
    aru_coords : pandas.DataFrame
        DataFrame with index filepath, and columns for x, y, (z) positions of recievers in meters.
        Third coordinate is optional. Localization algorithms are in 2d if columns are (x,y) and
        3d if columns are (x,y,z). Each audio file in `predictions` must have a corresponding
        row in `aru_coords` specifiying the position of the reciever.
    sample_rate : int
        Sample rate of the audio files
    min_number_of_receivers : int
        Minimum number of receivers that must detect an event for it to be localized
    max_distance_between_receivers : float (meters)
        Radius around a recorder in which to use other recorders for localizing an event
    localization_algorithm : str, optional
        algorithm to use for estimating the position of a sound event from the positions and
        time delays of a set of detections. [Default: 'gillette']
        Options:
            - 'gillette': linear closed-form algorithm of Gillette and Silverman 2008 [1]
            - 'soundfinder': source? citation? #TODO
    thresholds : dict, optional
        Dictionary of thresholds for each class. Default is None.
    bandpass_ranges : dict, optional
        Dictionary of form {"class name": [low_f, high_f]} for audio bandpass filtering during
        cross correlation. [Default: None] does not bandpass audio. Bandpassing audio to the
        frequency range of the relevant sound is recommended for best cross correlation results.
    max_delay : float, optional
        Maximum absolute value of time delay estimated during cross correlation of two signals
        For instance, 0.2 means that cross correlation will be maximized in the range of
        delays between -0.2 to 0.2 seconds.
        Default: None does not restrict the range, finding delay that maximizes cross correlation
    cc_threshold : float, optional
        Threshold for cross correlation: if the max value of the cross correlation is below
        this value, the corresponding time delay is discarded and not used during localization.
        Default of 0 does not discard any delays.
    cc_filter : str, optional
        Filter to use for generalized cross correlation. See signalprocessing.gcc function for options.
        Default is "phat".
    residual_threshold: discard localized events if the root mean squared residual exceeds this value
        (distance in meters)

    Methods
    -------
    localize()
        Run the enitre localization algorithm on the audio files and predictions. This executes the below methods in order.
        threshold_predictions()
            Use a set of score thresholds to filter the predictions and ensure that only detections with a minimum number of receivers are returned.
            Saves detections as a pandas.DataFrame to self.detections.
        cross_correlate()
            Cross correlate the audio files to get time delays of arrival. This is computationally expensive.
            Saves cross correlations as a pandas.DataFrame to self.cross_correlations.
        filter_cross_correlations()
            Filter the cross correlations to remove scores below cc_threshold. This then also ensures at least min_number_of_receivers are present.
            Saves filtered cross correlations as a pandas.DataFrame to self.filtered_cross_correlations.
        localize_events()
            Use the localization algorithm to localize the events from the set of tdoas after filtering.
            Saves locations as a pandas.DataFrame to self.localized_events.


    [1] M. D. Gillette and H. F. Silverman, "A Linear Closed-Form Algorithm for Source Localization From Time-Differences of Arrival," IEEE Signal Processing Letters

    """

    def __init__(
        self,
        aru_coords,
    ):
        self.aru_coords = aru_coords

        # attributes for troubleshooting
        self.files_missing_coordinates = []

        # check that all files have coordinates in aru_coords
        self.files = list(self.detections.reset_index()["file"].unique())
        audio_files_have_coordinates = True
        for file in self.files:
            if str(file) not in self.aru_coords.index:
                audio_files_have_coordinates = False
                self.files_missing_coordinates.append(file)
        if not audio_files_have_coordinates:
            raise UserWarning(
                "WARNING: Not all audio files have corresponding coordinates. Check aru_coords contains a mapping for each file. \n Check the missing files with Localizer.files_missing_coordinates"
            )
        # check that bandpass_ranges have been set for all classes
        if self.bandpass_ranges is not None:
            if set(self.bandpass_ranges.keys()) != set(self.predictions.columns):
                warnings.warn(
                    "WARNING: Not all classes have corresponding bandpass ranges. Default behavior will be to not bandpass before cross-correlation for classes that do not have a corresponding bandpass range."
                )  # TODO support one bandpass range for all classes

        print("SynchronizedRecorderArray initialized")

    def localize_detections(
        self,
        detections,
        min_number_of_receivers=3,
        max_distance_between_receivers=None,
        localization_algorithm="gillette",
        cc_threshold=0,
        cc_filter="phat",
        max_delay=None,
        bandpass_ranges=None,
        residual_threshold=np.inf,
    ):
        """
        Attempt to localize positions for all detections

        Args:
            detections: a dictionary of detections, with multi-index (file,start_time,end_time), and
                one column per class with 0/1 values for non-detection/detection
                The times in the index imply the same real world time across all files: eg 0 seconds assumes
                that the audio files all started at the same time, not on different dates/times

        Returns:
            2 lists: list of localized events, list of un-localized events
            events are of class SpatialEvent
        """
        # initialize list to store events that successfully localize
        localized_events = []
        unlocalized_events = []

        # create list of SpatialEvent objects to attempt to localize
        # creates events for every detection, adding nearby detections
        # to assist in localization via time delay of arrival
        candidate_events = self.create_candidate_events(
            detections,
            self.aru_coords,
            min_number_of_receivers,
            max_distance_between_receivers,
        )

        # attempt to localize each event
        for event in candidate_events:
            # choose bandpass range based on this event's detected class
            if bandpass_ranges is not None:
                bandpass_range = bandpass_ranges[event.class_name]
            else:
                bandpass_range = None

            # perform gcc to estiamte relative time of arrival at each receiver
            # relative to the first in the list (reference receiver)
            event.estimate_delays(
                bandpass_range=bandpass_range,
                cc_filter=cc_filter,
                max_delay=max_delay,
            )

            # estimate positions of sound event using time delays and receiver positions
            try:
                event.estimate_position(
                    algorithm=localization_algorithm,
                    cc_threshold=cc_threshold,
                    min_num_receivers=self.min_number_of_receivers,
                    speed_of_sound=SPEED_OF_SOUND,  # TODO make these into arguments of the function?
                )
            except InsufficientReceiversError:
                # this occurs if not enough receivers had high enough cross correlation scores
                # to continue with localization (<min_number_of_receivers)
                unlocalized_events.append(event)
                continue

            # check if residuals are small enough that we consider this a good position estimate
            # TODO: use max instead of mean?
            residual = event.calc_residual_rms()
            if residual < residual_threshold:
                localized_events.append(event)
            else:
                unlocalized_events.append(event)

        return localized_events, unlocalized_events

    def create_candidate_events(
        self,
        detections,
        aru_coords,
        min_number_of_receivers,
        max_distance_between_receivers,
    ):
        """
        Takes the detections dictionary and groups detections that are within max_distance_between_receivers of each other.
        args:
            detections: a dictionary of detections, with multi-index (file,start_time,end_time), and
                one column per class with 0/1 values for non-detection/detection
                The times in the index imply the same real world time across all files: eg 0 seconds assumes
                that the audio files all started at the same time, not on different dates/times
            aru_coords: a dictionary of aru coordinates, with key audio file path, and value (x,y) coordinates
                in meters
            max_distance_between_receivers: the maximum distance between recorders to consider a detection as a single event
        returns:
            a list of SpatialEvent objects to attempt to localize
        """
        # pre-generate a dictionary listing all close files for each audio file
        # dictionary will have a key for each audio file, and value listing all other receivers
        # within max_distance_between_receivers of that receiver
        #
        # eg {ARU_0.mp3: [ARU_1.mp3, ARU_2.mp3...], ARU_1... }
        nearby_files_dict = dict()

        aru_files = aru_coords.keys()
        for aru in aru_files:  # loop over the aru files
            pos_aru = np.array(aru_coords.loc[aru])  # position of receiver
            other_arus = np.array(aru_coords)
            distances = other_arus - pos_aru
            euclid_distances = [np.linalg.norm(d) for d in distances]

            # boolean mask for whether recorder is close enough
            mask = [d <= max_distance_between_receivers for d in euclid_distances]
            nearby_files_dict[aru] = list(aru_files[mask])
            # does it include itself?

        # generate SpatialEvents for each detection, if enough nearby
        # receivers also had a detection at the same time
        # each SpatialEvent object contains the time and class name of a
        # detected event, a set of receivers' audio files, and receiver positions
        candidate_events = []  # list of SpatialEvents to try to localize

        # iterate through all classes in detections dataframe
        for cls_i in detections.columns:

            # select one column: contains 0/1 for each file and clip time period
            # (index: (file,start_time,end_time), values: 0 or 1)
            det_cls = detections[[cls_i]]

            # filter to positive detections of this class
            det_cls = det_cls[det_cls[cls_i] > 0]

            # iterate through each clip start time in the df of detections
            # note: all clips with start_time 0 are assumed to start at the same real-world time!
            # eg, not two recordings from different dates or times
            # TODO: maybe use datetime objects instead of just a time like 0 seconds?
            for time_i, dets_at_time_i in det_cls.groupby(level=1):
                # select all detections that occurred at the same time
                files_w_dets = dets_at_time_i.reset_index()["file"].unique()

                # for each file with detection, check how many nearby recorders have a detection
                # at the same time
                for ref_receiver_audio in files_w_dets:
                    # check how many other detections are close enough to be detections of
                    # the same sound event
                    # first, use pre-created dictionary of nearby receivers for each audio file
                    close_receivers = nearby_files_dict[ref_receiver_audio]
                    # then, subset files with detections to those that are nearby
                    close_receivers_w_dets = [
                        f for f in files_w_dets if f in close_receivers
                    ]

                    # if enough receivers, create a SpatialEvent using this set of receivers
                    if len(close_receivers_w_dets) + 1 >= min_number_of_receivers:
                        receiver_files = [ref_receiver_audio] + close_receivers_w_dets
                        receiver_positions = [aru_coords[r] for r in receiver_files]
                        clip_end = (
                            dets_at_time_i.loc[ref_receiver_audio, time_i, :]
                            .reset_index()["end_time"]
                            .values[0]
                        )  # hacky? how to get correct clip duration?
                        duration = clip_end - time_i

                        # create a SpatialEvent
                        candidate_events.append(
                            SpatialEvent(
                                receiver_files=receiver_files,
                                receiver_positions=receiver_positions,
                                start_time=time_i,
                                duration=duration,
                                class_name=cls_i,
                                bandpass_range=self.bandpass_ranges["class"],
                                cc_threshold=self.cc_threshold,
                                max_delay=self.max_delay,
                                # TODO
                            )
                        )

        return candidate_events

    # def cross_correlate(self):
    #     """
    #     Cross correlate the audio files to get time delays of arrival for each time interval where a sound event was detected on at least min_number_of_receivers.
    #     Returns a pandas.DataFrame and writes it to self.cross_correlations. Warning: this is computationally expensive.
    #     The DataFrame has columns:
    #         time : (start, end) tuple of the detection time in seconds
    #         reference_file: the reference file for the detection
    #         other_files: list of the other files against which cross correlation will be performed
    #         species: the species of the detection
    #         cross_correlations: list of the maximum cross-correlation score for each pair of files
    #         time_delays: list of the time delays corresponding to the maximal cross-correlation for each pair of files
    #     """
    #     if self.bandpass_ranges is None:
    #         warnings.warn(
    #             "No bandpass range set. Default behavior will be to not bandpass the audio before cross-correlation."
    #         )

    #     if self.max_delay is None:
    #         warnings.warn(
    #             "No max delay set. Default behavior will be to allow for any delay between the audio files."
    #         )
    #     if self.detections is None:
    #         print("No detections exist - running threshold_predictions")
    #         self.threshold_predictions()

    # def filter_cross_correlations(self):
    #     """
    #     Filter the cross-correlations to only include those that are above a certain threshold. This step also drops any detections where less than min_number_of_receivers are above the threshold.
    #     Returns a pandas.DataFrame and writes it to self.filtered_cross_correlations.
    #     The DataFrame has columns:
    #         time : (start, end) tuple of the detection time in seconds
    #         reference_file: the reference file for the detection
    #         other_files: list of the other files against which cross correlation will be performed
    #         species: the species of the detection
    #         cross_correlations: list of the maximum cross-correlation score for each pair of files
    #         time_delays: list of the time delays corresponding to the maximal cross-correlation for each pair of files
    #     """
    #     if self.cross_correlations is None:
    #         print("No cross correlations exist - running cross_correlate")
    #         self.cross_correlate()
    #     # filter the cross-correlations
    #     above_threshold = [
    #         cc > self.cc_threshold
    #         for cc in self.cross_correlations["cross_correlations"]
    #     ]

    #     n_before = len(self.cross_correlations)  # number of rows before filtering

    #     filtered_ccs = []
    #     filtered_files = []
    #     filtered_tdoas = []
    #     for i in range(len(self.cross_correlations)):
    #         mask = above_threshold[i]
    #         cc = self.cross_correlations["cross_correlations"].iloc[i]
    #         other_files = np.array(self.cross_correlations["other_files"].iloc[i])
    #         tdoa = np.array(self.cross_correlations["time_delays"].iloc[i])

    #         filtered_ccs.append(cc[mask])
    #         filtered_files.append(other_files[mask])
    #         filtered_tdoas.append(tdoa[mask])

    #     filtered_cross_correlations = self.cross_correlations.copy()

    #     filtered_cross_correlations["cross_correlations"] = filtered_ccs
    #     filtered_cross_correlations["other_files"] = filtered_files
    #     filtered_cross_correlations["time_delays"] = filtered_tdoas

    #     # Filter by the cc scores. If less than min_number_of_receivers have cc_score above threshold, drop them.
    #     ccs = [
    #         np.array(scores)
    #         for scores in filtered_cross_correlations["cross_correlations"]
    #     ]
    #     num_ccs_above_threshold = [sum(a > self.cc_threshold) for a in ccs]
    #     mask = np.array(num_ccs_above_threshold) >= self.min_number_of_receivers - 1
    #     filtered_cross_correlations = filtered_cross_correlations[mask]

    #     n_after = len(filtered_cross_correlations)  # number of rows after filtering
    #     print(f"{n_before - n_after} rows deleted")
    #     self.filtered_cross_correlations = filtered_cross_correlations
    #     return filtered_cross_correlations

    # def localize_events(self):
    #     """
    #     Localize the events using the localization algorithm specified in self.localization_algorithm. Returns a pandas.DataFrame with the results and writes it to self.localizations
    #     The columns of the DataFrame are:
    #         time : (start, end) tuple of the detection time in seconds
    #         reference_file: the reference file for the detection
    #         other_files: list of the other files against which cross correlation will be performed
    #         species: the species of the detection
    #         cross_correlations: list of the maximum cross-correlation score for each pair of files
    #         time_delays: list of the time delays corresponding to the maximal cross-correlation for each pair of files
    #         predicted_x: the predicted x coordinate of the event
    #         predicted_y: the predicted y coordinate of the event
    #         predicted_z: the predicted z coordinate of the event
    #         tdoa_error: the residuals in the tdoas against what would be expected from the predicted location.
    #     """
    #     if self.filtered_cross_correlations is None:
    #         print(
    #             "No filtered cross_correlations exist - running filter_cross_correlations"
    #         )
    #         self.filter_cross_correlations()
    #     localized = self.filtered_cross_correlations.copy()
    #     locations = []

    #     for index, row in self.filtered_cross_correlations.iterrows():
    #         reference = row["reference_file"]
    #         others = row["other_files"]
    #         reference_coords = self.aru_coords.loc[reference]
    #         others_coords = [self.aru_coords.loc[i] for i in others]
    #         all_coords = [reference_coords] + others_coords
    #         # add 0 tdoa for reference receiver
    #         delays = np.insert(row["time_delays"], 0, 0)

    #         location = localize(
    #             all_coords, delays, algorithm=self.localization_algorithm
    #         )
    #         locations.append(location)
    #     localized["predicted_location"] = locations
    #     self.localized_events = localized
    #     return localized

    # def _get_cross_correlations(
    #     reference_file,
    #     other_files,
    #     start_time,
    #     end_time,
    #     bandpass_range,
    #     max_delay,
    #     SAMPLE_RATE,
    #     cc_filter,
    # ):
    #     """
    #     Gets the maximal cross correlations and the time-delay (in s) corresponding to that cross correlation between
    #     the reference_file and other_files. Setting max_delay ensures that only cross-correlations
    #     +/- a certain time-delay are returned. i.e if a sound can be a maximum of +/-
    #     ----
    #     args:
    #         reference_file: Path to reference file.
    #         other_files: List of paths to the other files which will be cross-correlated against reference_file
    #         start_time: start of time segment (in seconds) to be cross-correlated
    #         end_time: end of time segment (in seconds) to be cross-correlated.
    #         bandpass_range: [lower, higher] of bandpass range. If None, no bandpass filter is applied.
    #         max_delay: the maximum time (in seconds) to return cross_correlations for. i.e. if the best cross correlation
    #                     occurs for a time-delay greater than max_delay, the function will not return it, instead it will return
    #                     the maximal cross correlation within +/- max_delay
    #         SAMPLE_RATE: the sampling rate of the audio.
    #         cc_filter: the filter to use for cross-correlation. see signalprocessing.gcc for options. Options currently are "phat" or "cc"
    #     returns:
    #         ccs: list of maximal cross-correlations for each pair of files.
    #         time_differences: list of time differences (in seconds) that yield the maximal cross-correlation.
    #     """
    #     if bandpass_range is None:
    #         # no bandpass filter
    #         reference_audio = Audio.from_file(
    #             reference_file, offset=start_time, duration=end_time - start_time
    #         )
    #         other_audio = [
    #             Audio.from_file(i, offset=start_time, duration=end_time - start_time)
    #             for i in other_files
    #         ]
    #     else:
    #         lower = min(bandpass_range)
    #         higher = max(bandpass_range)

    #         reference_audio = Audio.from_file(
    #             reference_file, offset=start_time, duration=end_time - start_time
    #         ).bandpass(lower, higher, order=9)
    #         other_audio = [
    #             Audio.from_file(
    #                 i, offset=start_time, duration=end_time - start_time
    #             ).bandpass(lower, higher, order=9)
    #             for i in other_files
    #         ]
    #     ccs = np.zeros(len(other_audio))
    #     time_difference = np.zeros(len(other_audio))
    #     for index, audio_object in enumerate(other_audio):
    #         delay, cc = sp.tdoa(
    #             audio_object.samples,
    #             reference_audio.samples,
    #             cc_filter=cc_filter,
    #             sample_rate=SAMPLE_RATE,
    #             return_max=True,
    #             max_delay=max_delay,
    #         )

    #         time_difference[index] = delay
    #         ccs[index] = cc

    #     return ccs, time_difference

    # def _get_detections(predictions_df, cnn_score_threshold):
    #     """
    #     Takes the predictions_df of CNN scores *FOR A SINGLE SPECIES*, chooses only detections > cnn_score_threshold
    #     and outputs a dictionary of times at which events were detected, and the ARU files they were detected in.
    #     args:
    #         predictions_array: a dataframe with multi-index of (file, start_time, end_time) with a column that is values for model predictions
    #         *FOR A SINGLE SPECIES*
    #         cnn_score_threshold: the minimum CNN score needed for a time-window to be considered a detection.
    #     returns:
    #         A dictionary of predictions, with key (start_time, end_time), and value list of files with detection triggered
    #         e.g. {(0.0,2.0): [ARU_0.mp3. ARU_1.mp3]}
    #     """
    #     # get the detections from the predictions
    #     # Threshold the scores to above cnn_score_threshold
    #     booleans = (
    #         predictions_df.loc[:, :, :] > cnn_score_threshold
    #     )  # find rows above threshold
    #     indices = (
    #         booleans[booleans].dropna().index
    #     )  # choose just those rows. dropna required to drop the others
    #     recorders = indices.get_level_values(
    #         0
    #     )  # get the list of recorders out of the multi-index
    #     indices = indices.droplevel(level=0)  # drop the recorders

    #     dataframe = pd.DataFrame(
    #         data=recorders, index=indices
    #     )  # df with index (start_time, end_time)
    #     dataframe = (
    #         dataframe.sort_index()
    #     )  # done to ensure speed-up and not get performancewarning
    #     recorders_list = []
    #     for idx in dataframe.index.unique():
    #         recorders_in_time = dataframe.loc[idx].values
    #         recorders_in_time = [
    #             i[0] for i in recorders_in_time
    #         ]  # to get recorder path string out of numpy array
    #         recorders_list.append(recorders_in_time)
    #     return dict(zip(dataframe.index.unique(), recorders_list))


def calc_speed_of_sound(temperature=20):
    """
    Calculate speed of sound in air, in meters per second

    Calculate speed of sound for a given temperature
    in Celsius (Humidity has a negligible
    effect on speed of sound and so this functionality
    is not implemented)

    Args:
        temperature: ambient air temperature in Celsius

    Returns:
        the speed of sound in air in meters per second
    """
    return 331.3 * np.sqrt(1 + float(temperature) / 273.15)


def lorentz_ip(u, v=None):
    """
    Compute Lorentz inner product of two vectors

    For vectors `u` and `v`, the
    Lorentz inner product for 3-dimensional case is defined as

        u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3]

    Or, for 2-dimensional case as

        u[0]*v[0] + u[1]*v[1] - u[2]*v[2]

    Args:
        u: vector with shape either (3,) or (4,)
        v: vector with same shape as x1; if None (default), sets v = u

    Returns:
        float: value of Lorentz IP"""
    if v is None:
        v = u

    if len(u) == 3 and len(v) == 3:
        c = [1, 1, -1]
        return sum([u[i] * v[i] * c[i] for i in range(len(u))])
    elif len(u) == 4 and len(v) == 4:
        c = [1, 1, 1, -1]
        return sum([u[i] * v[i] * c[i] for i in range(len(u))])

    return ValueError(f"length of x should be 3 or 4, was{len(u)}")


def travel_time(source, receiver, speed_of_sound):
    """
    Calculate time required for sound to travel from a souce to a receiver

    Args:
        source: cartesian position [x,y] or [x,y,z] of sound source, in meters
        receiver: cartesian position [x,y] or [x,y,z] of sound receiver, in meters
        speed_of_sound: speed of sound in m/s

    Returns:
        time in seconds for sound to travel from source to receiver
    """
    distance = np.linalg.norm(np.array(source) - np.array(receiver))
    return distance / speed_of_sound


def localize(receiver_positions, tdoas, algorithm, speed_of_sound=SPEED_OF_SOUND):
    """
    Perform TDOA localization on a sound event.
    Args:
        receiver_positions: a list of [x,y,z] positions for each receiver
            Positions should be in meters, e.g., the UTM coordinate system.
        tdoas: a list of TDOA times (onset times) for each recorder
            The times should be in seconds.
        speed_of_sound: speed of sound in m/s
        algorithm: the algorithm to use for localization
            Options: 'soundfinder', 'gillette'
    Returns:
        The estimated source position in meters.
    """
    if algorithm == "soundfinder":
        estimate = soundfinder_localize(receiver_positions, tdoas, speed_of_sound)
    elif algorithm == "gillette":
        estimate = gillette_localize(receiver_positions, tdoas, speed_of_sound)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Implemented for 'soundfinder' and 'gillette'"
        )
    return estimate


def soundfinder_localize(
    receiver_positions,
    arrival_times,
    speed_of_sound=SPEED_OF_SOUND,
    invert_alg="gps",  # options: 'gps'
    center=True,  # True for original Sound Finder behavior
    pseudo=True,  # False for original Sound Finder
):

    """
    Use the soundfinder algorithm to perform TDOA localization on a sound event
    Localize a sound event given relative arrival times at multiple receivers.
    This function implements a localization algorithm from the
    equations described in the class handout ("Global Positioning
    Systems"). Localization can be performed in a global coordinate
    system in meters (i.e., UTM), or relative to recorder positions
    in meters.
    Args:
        receiver_positions: a list of [x,y,z] positions for each receiver
          Positions should be in meters, e.g., the UTM coordinate system.
        arrival_times: a list of TDOA times (onset times) for each recorder
          The times should be in seconds.
        speed of sound: speed of sound in m/s
        invert_alg: what inversion algorithm to use (only 'gps' is implemented)
        center: whether to center recorders before computing localization
          result. Computes localization relative to centered plot, then
          translates solution back to original recorder locations.
          (For behavior of original Sound Finder, use True)
        pseudo: whether to use the pseudorange error (True) or
          sum of squares discrepancy (False) to pick the solution to return
          (For behavior of original Sound Finder, use False. However,
          in initial tests, pseudorange error appears to perform better.)
    Returns:
        The solution (x,y,z) in meters.
    """
    # make sure our inputs follow consistent format
    receiver_positions = np.array(receiver_positions).astype("float64")
    arrival_times = np.array(arrival_times).astype("float64")

    # The number of dimensions in which to perform localization
    dim = receiver_positions.shape[1]

    ##### Shift coordinate system to center receivers around origin #####
    if center:
        warnings.warn("centering")
        p_mean = np.mean(receiver_positions, 0)
        receiver_positions = np.array([p - p_mean for p in receiver_positions])
    else:
        warnings.warn("not centering")

    ##### Compute B, a, and e #####
    # Find the pseudorange, rho, for each recorder
    # pseudorange (minus a constant) ~= distances from source to each receiver
    rho = np.array([arrival_times * (-1 * speed_of_sound)]).T

    # Concatenate the pseudorange column to form matrix B
    B = np.concatenate((receiver_positions, rho), axis=1)

    # Vector of ones
    e = np.ones(receiver_positions.shape[0])

    # The vector of squared Lorentz norms
    a = 0.5 * np.apply_along_axis(lorentz_ip, axis=1, arr=B)

    # choose between two algorithms to invert the matrix
    if invert_alg != "gps":
        raise NotImplementedError
        # original implementation of lstsq:
        # Compute B+ * a and B+ * e
        # using closest equivalent to R's solve(qr(B), e)
        # Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
        # Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

    else:  # invert_alg == 'gps' ('special' falls back to 'lstsq')
        ## Compute B+ = (B^T \* B)^(-1) \* B^T
        # B^T * B

        to_invert = np.matmul(B.T, B)

        try:
            inverted = np.linalg.inv(to_invert)

        except np.linalg.LinAlgError as err:
            # for 'gps' algorithm, simply fail
            # if invert_alg == "gps":
            warnings.warn("4")
            if "Singular matrix" in str(err):
                warnings.warn("5")
                warnings.warn(
                    "Singular matrix. Were recorders linear or on same plane? Exiting with NaN outputs",
                    UserWarning,
                )
                return [[np.nan]] * (dim)
            else:
                warnings.warn("6")
                raise

            # for 'special' algorithm: Fall back to lstsq algorithm
            # elif invert_alg == "special":  #
            #     warnings.warn("7")
            #     Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
            #     Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

        else:  # inversion of the matrix succeeded
            # Compute B+ * a and B+ * e
            Bplus = np.matmul(inverted, B.T)
            Bplus_a = np.matmul(Bplus, a)
            Bplus_e = np.matmul(Bplus, e)

    ###### Solve quadratic equation for lambda #####

    # Compute coefficients
    cA = lorentz_ip(Bplus_e)
    cB = 2 * (lorentz_ip(Bplus_e, Bplus_a) - 1)
    cC = lorentz_ip(Bplus_a)

    # Compute discriminant
    disc = cB**2 - 4 * cA * cC
    # If discriminant is negative, set to zero to ensure
    # we get an answer, albeit not a very good one
    if disc < 0:
        disc = 0
        warnings.warn(
            "Discriminant negative--set to zero. Solution may be inaccurate. Inspect final value of output array",
            UserWarning,
        )

    # Compute options for lambda
    lamb = (-cB + np.array([-1, 1]) * np.sqrt(disc)) / (2 * cA)

    # Find solution u0 and solution u1
    ale0 = np.add(a, lamb[0] * e)
    u0 = np.matmul(Bplus, ale0)
    ale1 = np.add(a, lamb[1] * e)
    u1 = np.matmul(Bplus, ale1)

    # print('Solution 1: {}'.format(u0))
    # print('Solution 2: {}'.format(u1))

    ##### Return the better solution #####

    # Re-translate points
    if center:
        shift = np.append(p_mean, 0)  # 0 for b=error, which we don't need to shift
        u0 += shift
        u1 += shift

    # Select and return quadratic solution
    if pseudo:
        # Return the solution with the lower error in pseudorange
        # (Error in pseudorange is the final value of the position/solution vector)
        if abs(u0[-1]) <= abs(u1[-1]):
            return u0[0:-1]  # drop the final value, which is the error
        else:
            return u1[0:-1]  # drop the final value, which is the error

    else:
        # use the sum of squares discrepancy to choose the solution
        # This was the return method used in the original Sound Finder,
        # but it gives worse performance

        # Compute sum of squares discrepancies for each solution
        s0 = float(np.sum((np.matmul(B, u0) - np.add(a, lamb[0] * e)) ** 2))
        s1 = float(np.sum((np.matmul(B, u1) - np.add(a, lamb[1] * e)) ** 2))

        # Return the solution with lower sum of squares discrepancy
        if s0 < s1:
            return u0[0:-1]  # drop the final value, which is the error
        else:
            return u1[0:-1]  # drop the final value, which is the error


def gillette_localize(receiver_positions, arrival_times, speed_of_sound=SPEED_OF_SOUND):
    """
    Uses the Gillette and Silverman [1] localization algorithm to localize a sound event from a set of TDOAs.
    Args:
        receiver_positions: a list of [x,y] or [x,y,z] positions for each receiver
            Positions should be in meters, e.g., the UTM coordinate system.
        arrival_times: a list of TDOA times (arrival times) for each receiver
            The times should be in seconds.
        speed_of_sound: speed of sound in m/s
    Returns:
        coords: a tuple of (x,y,z) coordinates of the sound source


    Algorithm from:
    [1] M. D. Gillette and H. F. Silverman, "A Linear Closed-Form Algorithm for Source Localization
    From Time-Differences of Arrival," IEEE Signal Processing Letters
    """

    # check that these delays are with reference to one receiver (the reference receiver).
    # We do this by checking that one of the arrival times is within float precision
    # of 0 (i.e. arrival at the reference)
    print(arrival_times)
    if not np.isclose(np.min(np.abs(arrival_times)), 0):
        raise ValueError(
            "Arrival times must be relative to a reference receiver. Therefore the minimum arrival"
            " time must be 0 (corresponding to arrival at the reference receiver) None of your "
            "TDOAs are zero. Please check your arrival_times."
        )

    # make sure our inputs follow consistent format
    receiver_positions = np.array(receiver_positions).astype("float64")
    arrival_times = np.array(arrival_times).astype("float64")

    # The number of dimensions in which to perform localization
    dim = receiver_positions.shape[1]

    # find which is the reference receiver and reorder, so reference receiver is first
    ref_receiver = np.argmin(arrival_times)
    ordered_receivers = np.roll(receiver_positions, -ref_receiver, axis=0)
    ordered_tdoas = np.roll(arrival_times, -ref_receiver, axis=0)

    # Gillette silverman solves Ax = w, where x is the solution vector, A is a matrix, and w is a vector
    # Matrix A according to Gillette and Silverman (2008)
    A = np.zeros((len(ordered_tdoas) - 1, dim + 1))
    for column in range(dim + 1):
        if column < dim:
            A[:, column] = ordered_receivers[0, column] - ordered_receivers[1:, column]
        elif column == dim:
            A[:, column] = ordered_tdoas[1:] * speed_of_sound

    # Vector w according to Gillette and Silverman (2008)
    # w = 1/2 (dm0^2 - xm^2 - ym^2 - zm^2 + x0^2 + y0^2 + z0^2)
    X02 = np.sum(ordered_receivers[0] ** 2)  # x0^2 + y0^2 + z0^2
    dmx = ordered_tdoas[1:] * speed_of_sound
    XM2 = np.sum(ordered_receivers**2, axis=1)[1:]

    vec_w = 0.5 * (dmx + X02 - XM2)

    answer = np.linalg.lstsq(A, vec_w.T, rcond=None)
    coords = answer[0][:dim]
    # pseudorange = answer[0][dim]
    # residuals = answer[1]

    return coords


def calculate_tdoa_residuals(
    receiver_positions, tdoas, position_estimate, speed_of_sound
):  # TODO rewrite tests after refactoring
    """
    Calculate the residual distances of the TDOA localization algorithm

    The residual represents the discrepancy between (difference in distance
    of each reciever to estimated position) and (observed tdoa), and has
    units of meters. Residuals are calculated as follows:

        expected = calculated time difference of arrival between reference and
            another receiver, based on the positions of the receivers and
            estimated event position
        observed = observed tdoas provided to localization algorithm

        residual time = expected - observed (in seconds)

        residual distance = speed of sound * residual time (in meters)

    Args:
        receiver_position: The list of coordinates (in m) of each receiver,
            as [x,y] for 2d or or [x,y,z] for 3d.
        tdoas: List of time delays of arival for the sound at each receiver,
            relative to the first receiver in the list (tdoas[0] should be 0)
        position_estimate: The estimated position of the sound, as (x,y) or (x,y,z) in meters
        speed_of_sound: The speed of sound in m/s

    Returns:
        np.array containing the residuals in units of meters, one per receiver
    """
    # ensure all are numpy arrays
    receiver_positions = np.array(receiver_positions)
    tdoas = np.array(tdoas)
    position_estimate = np.array(position_estimate)

    # Calculate the TDOA residuals

    # calculate time sound would take to travel from the estimated position
    # to each receiver (distance/speed=time)
    distances = [np.linalg.norm(r - position_estimate) for r in receiver_positions]
    travel_times = np.array(distances) / speed_of_sound

    # the expected time _difference_ of arrival for any receiver vs the
    # reference receiver is the difference in travel times from the
    # position estimate to each of the receivers compared to the first
    expected_tdoas = travel_times - travel_times[0]

    # the time residual is the difference between the observed tdoa values
    # and those expected according to the estimated position
    # first value will be 0 by definition
    time_residuals = expected_tdoas - tdoas

    # convert residuals from units of time (s) to distance (m) via speed of sound
    return time_residuals * speed_of_sound
