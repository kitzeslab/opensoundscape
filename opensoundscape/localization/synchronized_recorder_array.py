import warnings
import numpy as np
import datetime
import pandas as pd

from opensoundscape.audio import Audio, parse_metadata
from opensoundscape.utils import cast_np_to_native
from opensoundscape.localization.spatial_event import (
    SpatialEvent,
    localize_events_parallel,
)
from opensoundscape.localization.localization_algorithms import SPEED_OF_SOUND


class SynchronizedRecorderArray:
    """
    Class with utilities for localizing sound events from array of recorders

    Methods
    -------
    localize_detections()
        Attempt to localize a sound event for each detection of each class.
        First, creates candidate events with:
        create_candidate_events()
            Create SpatialEvent objects for all simultaneous, spatially clustered detections of a class

        Then, attempts to localize each candidate event via time delay of arrival information:
        For each candidate event:
            - calculate relative time of arrival with generalized cross correlation (event.estimate_delays())
            - if enough cross correlation values exceed a threshold, attempt to localize the event
                using the time delays and spatial locations of each receiver with event.estimate_location()
            - if the residual distance rms value is below a cutoff threshold, consider the event
                to be successfully localized
    """

    def __init__(
        self,
        receiver_positions,
        file_receiver_map,
        speed_of_sound=SPEED_OF_SOUND,
    ):
        """
        Args:
            receiver_positions : pandas.DataFrame
                DataFrame with columns for x, y, (z) locations of each receiver in meters.
                Index is receiver ID or name.
                z coordinate is optional, if not provided localization is performed in 2D.
            file_receiver_map : dict
                a dictionary or pd.Series mapping from audio file path to receiver ID/name
                (index of receiver_positions)
            speed_of_sound : float, optional. Speed of sound in meters per second.
                Default: opensoundscape.localization.localization_algorithms.SPEED_OF_SOUND
        """
        if isinstance(file_receiver_map, pd.Series):  # allow pd.Series as input
            file_receiver_map = file_receiver_map.to_dict()
        self.receiver_positions = receiver_positions
        self.file_receiver = file_receiver_map
        self.speed_of_sound = speed_of_sound

    @classmethod
    def from_file_coords(
        cls,
        file_coords,
        speed_of_sound=SPEED_OF_SOUND,
    ):
        """
        Create a SynchronizedRecorderArray from a DataFrame of file coordinates.

        This function creates receiver_positions and file_receiver_map from file_coords.
        The receivers are named by an integer index.

        This function is provided for convenience and backwards compatability, but
        using receiver_positions and file_receiver_map directly is recommended, as
        it retains interpretable receiver names/IDs.

        Args:
            file_coords : pandas.DataFrame
                DataFrame with index filepath, and columns for x, y, (z) locations of receiver
                that recorded the audio file, in meters.
                z coordinate is optional, if not provided localization is performed in 2D.
            speed_of_sound : float, optional. Speed of sound in meters per second.
                Default: opensoundscape.localization.localization_algorithms.SPEED_OF_SOUND

        Returns:
            SynchronizedRecorderArray
        """
        receiver_positions, file_receiver_map = get_receiver_positions(file_coords)
        return cls(
            receiver_positions=receiver_positions,
            file_receiver_map=file_receiver_map,
            speed_of_sound=speed_of_sound,
        )

    def localize_detections(
        self,
        detections,
        max_receiver_dist,
        localization_algorithm="least_squares",
        max_delay=None,
        min_n_receivers=3,
        cc_threshold=0,
        cc_filter="phat",
        bandpass_ranges=None,
        residual_threshold=np.inf,
        return_unlocalized=False,
        num_workers=1,
    ):
        """
        Attempt to localize positions of sound sources from clip-level detections across multiple recorders

        Wraps self.create_candidate_events() and self.localize_events()

        Algorithm
        ----------
        The user provides a table of class detections from each recorder with timestamps. The
        object's self.receiver_positions dataframe contains a table listing the spatial location of
        each recorder, and the self.file_receiver_map dataframe contains a mapping of audio files to
        their corresponding recorders. The audio recordings must be synchronized such that
        timestamps from each recording correspond to the exact same real-world time.

        Localization of sound events proceeds in four steps:

        1. Grouping of detections into candidate events (self.create_candidate_events()):

            Simultaneous and spatially clustered detections of a class are selected as targets for
            localization of a single real-world sound event.

            For each detection of a species, the grouping algorithm treats the reciever with the
            detection as a "reference receiver", then selects all detections of the species at the
            same time and within `max_receiver_dist` of the reference reciever (the "surrounding
            detections"). This selected group of simulatneous, spatially-clustered detections of a
            class beomes one "candidate event" for subsequent localization.

            If the number of recorders in the candidate event is fewer than `min_n_receivers`, the
            candidate event is discarded.

            This step creates a highly redundant set of candidate events to localize, because each
            detection is treated separately with its recorder as the 'reference recorder'. Thus, the
            localized events created by this algorithm may contain multiple instances representing
            the same real-world sound event.


        2. Estimate time delays with cross correlation:

            For each candidate event, the time delay between the reference reciever's detection and
            the surrounding recorders' detections is estimated through generalized cross
            correlation.

            If bandpass_ranges are provided, cross correlation is performed on audio that has been
            bandpassed to class-specific low and high frequencies.

            If the max value of the cross correlation is below `cc_threshold`, the corresponding
            time delay is discarded and not used during localization. This provides a way of
            filtering out undesired time delays that do not correspond to two recordings of the same
            sound event.

            If the number of estimated time delays in the candidate event is fewer than
            `min_n_receivers` after filtering by cross correlation threshold, the candidate event is
            discarded.

        3. Estimate locations

            The location of the event is estimated based on the locations and time delays of each
            detection.

            location estimation from the locations and time delays at a set of receivers is
            performed using one of two algorithms, described in `localization_algorithm` below.

        4. Filter by spatial residual error

            The residual errors represent descrepencies between (a) time of arrival of the event at
            a reciever and (b) distance from reciever to estimated location.

            Estimated locations are discarded if the root mean squared spatial residual is greater
            than `residual_rms_threshold`


        Args:
            detections: a dataframe of sound event detections with multi-index (file,start_time,end_time),
                and one column per class with 0/1 values for non-detection/detection

                The times in the index imply the same real world time across all files. E.g.,
                 0 seconds assumes that the audio files all started at the same time, not on
                    different dates/times

            max_receiver_dist : float (meters)
                Radius around a recorder in which to use other recorders for localizing an event.
                Simultaneous detections at receivers within this distance (meters) of a receiver
                with a detection will be used to attempt to localize the event.

            max_delay : float, optional
                Maximum absolute value of time delay estimated during cross correlation of two
                signals For instance, 0.2 means that the maximal cross-correlation in the range of
                delays between -0.2 to 0.2 seconds will be used to estimate the time delay. if None
                (default), the max delay is set to max_receiver_dist / self.speed_of_sound

            min_n_receivers : int
                Minimum number of receivers that must detect an event for it to be localized
                [default: 3]

            localization_algorithm : str, optional
                algorithm to use for estimating the location of a sound event from the locations and
                time delays of a set of detections. [Default: 'least_squares'] Options:
                    - 'least_squares': least squares optimization [default]
                    - 'gillette': linear closed-form algorithm of Gillette and Silverman 2008 [1]
                    - 'soundfinder': GPS location algorithm of Wilson et al. 2014 [2]
                    - 'least_squares': nonlinear least squares minimization of residuals

            cc_threshold : float, optional
                Threshold for cross correlation: if the max value of the cross correlation is below
                this value, the corresponding time delay is discarded and not used during
                localization. Default of 0 does not discard any delays.

            cc_filter : str, optional
                Filter to use for generalized cross correlation. See signalprocessing.gcc function
                for options. Default is "phat".

            bandpass_ranges : dict, None, or 2-element list/tuple, optional. Any of:
                - dict: {"class name": [low_f, high_f]} for audio bandpass filtering during
                cross correlation with keys for each class - list/tuple: [low_f,high_f] for all
                classes - None [Default]: does not bandpass audio.

                Note: Bandpassing audio to the frequency range of the relevant sound is recommended
                for best cross correlation results.

            residual_threshold: discard localized events if the root mean squared residual of the
            TDOAs exceeds this value (a distance in meters) [default: np.inf does not filter out any
            events by residual]

            return_unlocalized: bool, optional. If True, returns a second value, the list of
                PositionEstimates that either failed to localize or was filtered out, for example
                because too few receivers had detections, or too few receivers passed the
                cc_threshold, or the TDOA residuals were too high.

            num_workers : int, optional. Number of workers to use for parallelization. Default is 1
                (no parallelization, no multiprocessing). If > 1, uses joblib.Parallel to parallelize

        Returns:
            If return_unlocalized is False,
                returns a list of localized positons, each of which is a PositionEstimate object
                with a .location_estimate attribute (= None if localization failed)
            If return_unlocalized is True, returns 2 lists:
                list of localized positions, list of un-localized positions

        [1] M. D. Gillette and H. F. Silverman, "A Linear Closed-Form Algorithm for Source
        Localization From Time-Differences of Arrival," IEEE Signal Processing Letters

        [2]  Wilson, David R., Matthew Battiston, John Brzustowski, and Daniel J. Mennill. “Sound
        Finder: A New Software Approach for Localizing Animals Recorded with a Microphone Array.”
        Bioacoustics 23, no. 2 (May 4, 2014): 99–112. https://doi.org/10.1080/09524622.2013.827588.
        """

        # create list of SpatialEvents, each SpatialEvent will be used to estimate a location
        # each SpatialEvent consists of a receiver with a detection, and every other receivers within max_receiver_dist, that also have a detection
        # TDOA estimation and localization will be performed on each SpatialEvent
        # multiple SpatialEvents may refer to the same real-world sound event
        candidate_events = self.create_candidate_events(
            detections,
            min_n_receivers,
            max_receiver_dist,
            cc_threshold=cc_threshold,
            bandpass_ranges=bandpass_ranges,
            cc_filter=cc_filter,
            max_delay=max_delay,
        )

        # localize each event with joblib.Parallel for parallelization
        # get back list of PositionEstimate objects
        position_estimates = localize_events_parallel(
            events=candidate_events,
            num_workers=num_workers,
            localization_algorithm=localization_algorithm,
        )

        # separate positions into localized and unlocalized:

        # PositionEstimates for events that we do not consider valid localizations (e.g. too few
        # receivers, too few receivers after applying cc_threshold or high residual in localization)
        unlocalized_positions = [
            e
            for e in position_estimates
            if (e.location_estimate is None or e.residual_rms > residual_threshold)
        ]

        # list of PositionEstimates for events that were successfully localized and passed filters
        localized_positions = [
            e
            for e in position_estimates
            if (
                e.location_estimate is not None and e.residual_rms <= residual_threshold
            )
        ]

        # return_unlocalized can be used for troubleshooting, and working out why some events were
        # not localized, but typically, we just want the PositionEstimates that were successfully
        # localized
        if return_unlocalized:
            return localized_positions, unlocalized_positions
        else:
            return localized_positions

    def localize_events_msrp(
        self,
        events,
        resolution,
        margin,
        audio_sample_rate=None,
        spatial_grid=None,
        num_workers=1,
        keep_power_map=False,
        **kwargs,
    ):
        """Localize events in parallel using the modified steered-response power (M-SRP) algorithm.

        For each SpatialEvent in `events`, this function will extract aligned audio
        segments from the event's receiver files and call `msrp.localize()` for
        each event in parallel.

        Creates a `SearchMap` (if `spatial_grid` is not provided), computes the valid time intervals
        of the search map if not pre-computed.

        Args:
            events (list[SpatialEvent]): candidate events to localize.
            resolution (float): grid resolution in meters for SearchMap creation if
                `spatial_grid` is not provided.
            margin (float): margin in meters to expand the convex hull when
                creating the SearchMap.
            audio_sample_rate (float): sample rate of the audio (required if
                creating a SearchMap here).
            spatial_grid (SearchMap or None): optional precomputed SearchMap to use.
            num_workers (int): number of parallel workers.
            keep_power_map (bool): if True, include 'power_map' and 'search_map'
                in returned PositionEstimate objects.
            **kwargs: forwarded to `msrp.localize()` (e.g. freq_low, freq_high,
                cc_filter, aggregation_fn, convex_hull_margin, detrend).

        Returns:
            list[PositionEstimate]: list of position estimates (one per event). If
            `keep_power_map` is True, each PositionEstimate will include `power_map`
            (pd.Series) and `search_map` attributes.

        See also:
            SynchronizedRecorderArray.localize_detections()
        """

        from opensoundscape.localization import msrp

        if spatial_grid is None:
            # generate the grid and time delay intervals on which to evaluate steered response power
            assert (
                audio_sample_rate is not None
            ), "must provide audio_sample_rate for msrp localization if not providing spatial_grid"

            spatial_grid = msrp.SearchMap(
                receiver_positions=self.receiver_positions,
                sample_rate=audio_sample_rate,
                resolution=resolution,
                margin=margin,
                speed_of_sound=self.speed_of_sound,
                compute_time_intervals=True,
            )

        return localize_events_parallel(
            events=events,
            num_workers=num_workers,
            localization_algorithm="msrp",
            search_map=spatial_grid,
            keep_power_map=keep_power_map,
            **kwargs,
        )

    def localize_detections_msrp(
        self,
        detections,
        max_receiver_dist,
        audio_sample_rate,
        resolution=1,
        margin=0,
        min_n_receivers=3,
        cc_filter="phat",
        bandpass_ranges=None,
        spatial_grid=None,
        num_workers=1,
        keep_power_map=False,
        **kwargs,
    ):
        """Localize detections using the modified steered-response power algorithm.

        This convenience wrapper groups detections into candidate SpatialEvents
        (via `create_candidate_events`) and then localizes them using
        `localize_events_msrp`.
        Args:
            detections (pd.DataFrame): multi-index DataFrame of detections
                (index: file, start_time, end_time, start_timestamp; columns:
                class names with 0/1 values for non-detection/detection).
            max_receiver_dist (float): maximum distance in meters between receivers.
            audio_sample_rate (float): sample rate of the audio files.
            resolution (float): grid resolution in meters for SearchMap creation
                if `spatial_grid` is not provided.
            margin (float): margin in meters to expand the convex hull when
                creating the SearchMap.
            min_n_receivers (int): minimum number of receivers that must detect an
                event for it to be localized. [default: 3]
            cc_filter (str): filter to use for generalized cross correlation. See
                opensoundscape.signal_processing.gcc() function for options. Default is "phat".
            bandpass_ranges (dict, None, or 2-element list/tuple, optional): Any of:
                - dict: {"class name": [low_f, high_f]} for audio bandpass filtering during
                cross correlation with keys for each class
                - list/tuple: [low_f,high_f] for all classes
                - None [Default]: does not bandpass audio.
            spatial_grid (SearchMap or None): optional precomputed SearchMap to use.
            num_workers (int): number of parallel workers.
            keep_power_map (bool): if True, include 'power_map' and 'search_map'
                in returned PositionEstimate objects.
            **kwargs: forwarded to `msrp.localize()` (e.g. freq_low, freq_high,
                cc_filter, aggregation_fn, convex_hull_margin, detrend).

        Returns:
            list[PositionEstimate]: localized position estimates. If
            `keep_power_map` is True, returned PositionEstimate objects will
            include `power_map` and `search_map` attributes.
        """
        # create list of SpatialEvents, each SpatialEvent will be used to estimate a location
        # each SpatialEvent consists of a receiver with a detection, and every other receivers within max_receiver_dist
        # localization will be performed on each SpatialEvent
        # multiple SpatialEvents may refer to the same real-world sound event
        candidate_events = self.create_candidate_events(
            detections,
            min_n_receivers,
            max_receiver_dist,
            cc_filter=cc_filter,
            bandpass_ranges=bandpass_ranges,
            cc_threshold=0,  # not used for msrp
        )

        # localize each event with joblib.Parallel for parallelization
        # get back list of PositionEstimate objects
        position_estimates = self.localize_events_msrp(
            events=candidate_events,
            resolution=resolution,
            margin=margin,
            audio_sample_rate=audio_sample_rate,
            spatial_grid=spatial_grid,
            num_workers=num_workers,
            keep_power_map=keep_power_map,
            **kwargs,
        )

        return position_estimates

    def check_files_missing_coordinates(self, detections):
        """
        Check that all files in detections have a mapping to a receiver with coordinates

        Returns:
            - a list of files that are in detections but
                either (a) not in self.file_receiver_map or
                (b) mapped to a receiver ID not in self.receiver_positions
        """
        files_missing_coordinates = []
        files = list(detections.reset_index()["file"].unique())
        for file in files:
            if str(file) not in self.file_receiver:
                files_missing_coordinates.append(file)
            else:
                receiver_id = self.file_receiver[str(file)]
                if receiver_id not in self.receiver_positions.index:
                    # receiver ID not in receiver_positions df
                    files_missing_coordinates.append(file)
        return files_missing_coordinates

    def create_candidate_events(
        self,
        detections,
        min_n_receivers,
        max_receiver_dist,
        cc_threshold,
        bandpass_ranges,
        cc_filter,
        max_delay=None,
    ):
        """
        Takes the detections dictionary and groups detections that are within `max_receiver_dist` of each other.

        Args:
            detections: a dictionary of detections, with multi-index (file,start_time,end_time,start_timestamp), and
                one column per class with 0/1 values for non-detection/detection
                where start_timestamp contains timzeone-aware datetime.datetime objects corresponding to start_time
                - If `start_timestamp` index not present, attempts to automatically determine the
                start_timestamps file from audio file metadata ('recording_start_time' field).
                This will generally succeed with files generated by AudioMoth or other devices for which metadata parsing
                is supported and recording start time is given in the metadata.

                Example of adding `start_timestamp` to multi-index of detections df manually:
                ```python
                # assume we have a csv with columns file, start_time, end_time, class1, ...
                # e.g., the output of and opensoundscape CNN prediction workflow
                detections = pd.read_csv('my_opso_detections.csv',index_col=[0,1,2])
                # let's assume all files started recording at the same time for this example:
                import pytz
                import datetime
                tz = pytz.timezone('America/New_York')
                detections = detections.reset_index(drop=False)
                detections['start_timestamp']=[
                    datetime.datetime(2024,1,1,10,0,0).astimezonoe(tz)+
                    datetime.timedelta(detections.at[i,'start_time'])
                    for i in detections.index
                ]
                # add the start_timestamp column into the multi-index
                detections = detections.reset_index(drop=False).set_index(
                    ["file", "start_time", "end_time", "start_timestamp"]
                )
                ```

            min_n_receivers: if fewer nearby receivers have a simultaneous detection, do not create candidate event

            max_receiver_dist: the maximum distance between recorders to consider a detection as a single event

            bandpass_ranges: dictionary of bandpass ranges for each class, or a single range for all classes, or None
                - if None, does not bandpass audio before cross-correlation
                - if a single range [low_f,high_f], uses this bandpass range for SpatialEvents of all classes
                - if a dictionary, must have keys for each class in detections.columns

            max_delay: the maximum delay (in seconds) to consider between receivers for a single event
                if None, defaults to max_receiver_dist / self.speed_of_sound
        Returns:
            a list of SpatialEvent objects to attempt to localize
        """
        # check that all files have receiver mapping, and receivers have coordinates
        if len(self.check_files_missing_coordinates(detections)) > 0:
            warnings.warn(
                "WARNING: Not all audio files have corresponding receivers with coordinates."
                "Check self.file_receiver_map contains each file in detections.index, "
                "and that receiver IDs are mapped to coordinates in self.receiver_positions."
                "Use self.check_files_missing_coordinates() for list of files. "
            )

        # bandpass_range: for convenience, copy None or a single range to all classes
        if bandpass_ranges is None:
            bandpass_ranges = {cls: None for cls in detections.columns}
        if isinstance(bandpass_ranges, (list, tuple)):
            bandpass_ranges = {cls: bandpass_ranges for cls in detections.columns}

        # check that bandpass_ranges have been set for all classes
        if len(set(detections.columns) - set(bandpass_ranges.keys())) > 0:
            warnings.warn(
                "WARNING: Not all classes have corresponding bandpass ranges. "
                "Default behavior will be to not bandpass before cross-correlation for "
                "classes that do not have a corresponding bandpass range."
            )

        # max_delay in seconds determines how much audio is cross correlated, and
        # restricts the range of time delays to consider
        # if not given, calculate from speed of sound and radius of included receivers:
        # can't be longer than the time it takes sound to travel from the furthest receiver
        # to the reference receiver
        if max_delay is None:
            max_delay = max_receiver_dist / self.speed_of_sound

        # pre-generate a dictionary listing all close files for each audio file
        # dictionary will have a key for each audio file, and value listing all other receivers
        # within max_receiver_dist of that receiver
        #
        # eg {ARU_0.mp3: [ARU_1.mp3, ARU_2.mp3...], ARU_1... }
        # nearby_files_dict = self.make_nearby_files_dict(max_receiver_dist)
        nearby_receivers_dict = self.make_nearby_receivers_dict(max_receiver_dist)

        # generate SpatialEvents for each detection, if enough nearby
        # receivers also had a detection at the same time
        # each SpatialEvent object contains the time and class name of a
        # detected event, a set of receivers' audio files, and receiver locations
        # and represents a single sound event that we will try to localize
        #
        # events will be redundant because each reciever with detection potentially
        # results in its own event containing nearby detections
        candidate_events = []  # list of SpatialEvents to try to localize

        detections = detections.copy()  # don't change original

        if "start_timestamp" in detections.index.names:
            # check that the timestamps are localized datetime.datetime objects
            dts = detections.index.get_level_values("start_timestamp")

            def is_localized_dt(dt):
                # is it a datetime.datetime object?
                if not isinstance(dt, datetime.datetime):
                    return False
                # is it localized to a timezone?
                if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                    return False
                return True

            if not all([is_localized_dt(dt) for dt in dts]):
                raise ValueError(
                    "start_timestamp index must contain timezone-localized datetime.datetime objects, but "
                    "at least one value in the multi-index 'start_timestamp' column is not a localized datetime.datetime object. "
                    "See SynchronizedRecorderArray.create_candidate_events() docstring for example."
                )
        else:
            # try to determine start_timestamps from metadata of each audio file
            try:
                # initialize instance of helper class used to calculate timestamp from file metadata
                # class rather than function to allow caching and re-using timestamp when file remains the same
                get_start_ts = GetStartTimestamp()
                detections["start_timestamp"] = [
                    get_start_ts(file, start_time)
                    for (file, start_time, _) in detections.index
                ]
                # add `start_timestamp` to multi-index
                detections = detections.reset_index(drop=False).set_index(
                    ["file", "start_time", "end_time", "start_timestamp"]
                )
            except Exception as e:
                raise ValueError(
                    "could not determine `start_timestamp`s from audio file metadata. Include `start_timestamp` in the multi-index of detections. "
                    "See SynchronizedRecorderArray.create_candidate_events() docstring for example."
                ) from e

        # iterate through all classes in detections (0/1) dataframe
        # with index (file,start_time,end_time), column for each class
        for cls_i in detections.columns:
            if bandpass_ranges is not None:
                bandpass_range = bandpass_ranges[cls_i]
            else:
                bandpass_range = None
            # select one column: contains 0/1 for each file and clip time period
            # (index: (file,start_time,end_time), values: 0 or 1) for a single class
            det_cls = detections[[cls_i]]

            # filter detection dataframe to select detections of this class
            det_cls = det_cls[det_cls[cls_i] > 0]

            # iterate through each clip start_timestamp in the df of detections
            for timestamp_i, dets_at_time_i in det_cls.groupby(level="start_timestamp"):
                if len(dets_at_time_i) < min_n_receivers:
                    continue

                # list all files with detections of this class at the same time
                files_w_dets = dets_at_time_i.reset_index()["file"].unique()

                # for each file with detection of this class at this time,
                # check how many nearby recorders have a detection
                # at the same time. If there are enough, make a SpatialEvent
                # containing the spatial cluster of detections. The event
                # will be added to the list of candidate_events to localize
                for ref_file in files_w_dets:
                    # check how many other detections are close enough to be detections of
                    # the same sound event
                    # first, look up close receivers from pre-created dictionary
                    ref_receiver = self.file_receiver[ref_file]
                    close_receivers = nearby_receivers_dict[ref_receiver]
                    # then, subset files with detections to those that are nearby
                    close_det_files = [
                        f
                        for f in files_w_dets
                        if self.file_receiver[f] in close_receivers
                    ]

                    # if enough receivers, create a SpatialEvent using this set of receivers
                    # +1 to count the reference receiver
                    if len(close_det_files) + 1 >= min_n_receivers:
                        # SpatialEvent will include audio from reference receiver + close receivers
                        receiver_files = [ref_file] + close_det_files
                        receivers = [self.file_receiver[r] for r in receiver_files]

                        # retrieve positions of each receiver
                        receiver_locations = self.receiver_positions.loc[
                            receivers
                        ].values

                        # subset detections to close recievers, and re-index to only 'file'
                        close_dets = (
                            dets_at_time_i.reset_index()
                            .set_index("file")
                            .loc[receiver_files]
                        )

                        duration = (
                            close_dets.at[ref_file, "end_time"]
                            - close_dets.at[ref_file, "start_time"]
                        )

                        # create a SpatialEvent for this cluster of simultaneous detections
                        candidate_events.append(
                            SpatialEvent(
                                receivers=receivers,
                                receiver_files=receiver_files,
                                receiver_locations=receiver_locations,
                                bandpass_range=bandpass_range,
                                cc_threshold=cc_threshold,
                                max_delay=max_delay,
                                min_n_receivers=min_n_receivers,
                                # find the start_time value for each clip, i.e. offset from the start of
                                # the corresponding file to the start of the detection
                                # if all audio files started at same time, this will be the same for all files
                                receiver_start_time_offsets=close_dets[
                                    "start_time"
                                ].values,
                                start_timestamp=timestamp_i,
                                duration=duration,
                                class_name=cls_i,
                                cc_filter=cc_filter,
                                speed_of_sound=self.speed_of_sound,
                            )
                        )

        return candidate_events

    def make_nearby_receivers_dict(self, r_max):
        """create dictinoary listing nearby receivers for each receiver

        pre-generate a dictionary listing all close receivers for each receiver
        dictionary will have a key for each receiver ID/name, and value listing all other receivers
        within r_max of that receiver

        eg {rec_0: [rec_1, rec_2...], rec_1... }

        The returned dictionary is used in create_candidate_events as a look-up table for
            creating SpatialEvents with recordings in the vicinity of a detection
        """

        nearby_receivers_dict = dict()
        # make an entry in the dictionary for each recorder
        for rec_id in self.receiver_positions.index:
            reference_receiver = self.receiver_positions.loc[rec_id]
            other_receivers = self.receiver_positions.drop([rec_id])
            # compute euclidean distances to all other receivers
            euclid_distances = np.linalg.norm(
                other_receivers.values - reference_receiver.values, axis=1
            )

            # boolean mask for whether recorder is close enough
            mask = [r <= r_max for r in euclid_distances]

            # dictionary entry mapping ref receiver to other nearby receivers
            nearby_receivers_dict[rec_id] = list(other_receivers[mask].index)

        return nearby_receivers_dict


class GetStartTimestamp:
    def __init__(self) -> None:
        # cache one file's timestamp to avoid repeated metadata parsing
        # if fetching timestamps for multiple detections from the same file
        self.cached_file = ""
        self.cached_recording_start_time = None

    def __call__(self, file, start_time):
        """extract start_timestamp from file metadata, return start_timestamp + start_time in seconds

        Args:
            file: str, path to audio file
            start_time: float, time in seconds from start of file

        Returns:
            datetime.datetime: start timestamp parsed from audio file metadata + start_time
        """
        try:
            if file == self.cached_file:
                recording_start_dt = self.cached_recording_start_time
            else:
                recording_start_dt = parse_metadata(file)["recording_start_time"]
                self.cached_file = file
                self.cached_recording_start_time = recording_start_dt
        except Exception as exc:
            raise Exception(
                f"Failed to retrieve recording start time from metadata of file {file}."
            ) from exc

        return recording_start_dt + datetime.timedelta(
            seconds=cast_np_to_native(start_time)
        )


def get_receiver_positions(file_coords):
    """
    Given a DataFrame with file paths as index and columns ['x', 'y'] or ['x', 'y', 'z'],
    returns:
      - receiver_positions: DataFrame of unique receiver locations
      - file_to_recorder_idx: dict mapping file path -> row index of receiver_positions
    """

    # Ensure coordinate columns exist
    coord_cols = [c for c in ["x", "y", "z"] if c in file_coords.columns]
    if not coord_cols:
        raise ValueError("file_coords must have at least 'x' and 'y' columns")

    indices, coords = pd.factorize(pd.Series([tuple(x) for x in file_coords.values]))
    receiver_positions = pd.DataFrame(
        [list(t) for t in coords], columns=file_coords.columns
    )
    file_to_receiver_idx = {f: i for f, i in zip(file_coords.index, indices)}

    return receiver_positions, file_to_receiver_idx
