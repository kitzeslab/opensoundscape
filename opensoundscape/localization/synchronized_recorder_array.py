import warnings
import numpy as np
import datetime

from opensoundscape.audio import Audio
from opensoundscape.utils import cast_np_to_native
from opensoundscape.localization.spatial_event import (
    SpatialEvent,
    localize_events_parallel,
)
from opensoundscape.localization.localization_algorithms import SPEED_OF_SOUND


class GetStartTimestamp:
    def __init__(self) -> None:
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
                recording_start_dt = Audio.from_file(file, duration=0.0001).metadata[
                    "recording_start_time"
                ]  # TODO: replace with audio.metadata_from_file after merging develop
                self.cached_file = file
                self.cached_recording_start_time = recording_start_dt
        except Exception as exc:
            raise Exception(
                f"Failed to retrieve recording start time from metadata of file {file}."
            ) from exc

        return recording_start_dt + datetime.timedelta(
            seconds=cast_np_to_native(start_time)
        )


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
        file_coords,
        speed_of_sound=SPEED_OF_SOUND,
    ):
        """
        Args:
            file_coords : pandas.DataFrame
                DataFrame with index filepath, and columns for x, y, (z) locations of reciever that
                recorded the audio file, in meters.
                Third coordinate is optional. Localization algorithms are in 2d if columns are (x,y) and
                3d if columns are (x,y,z). When running .localize_detections() or .create_candidate_events(),
                Each audio file in `detections` must have a corresponding
                row in `file_coords` specifiying the location of the reciever that recorded the file.
            speed_of_sound : float, optional. Speed of sound in meters per second.
                Default: opensoundscape.localization.localization_algorithms.SPEED_OF_SOUND
        """
        self.file_coords = file_coords
        self.speed_of_sound = speed_of_sound

    def localize_detections(
        self,
        detections,
        max_receiver_dist,
        localization_algorithm="gillette",
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
        object's self.file_coords dataframe contains a table listing the spatial location of the
        recorder for each unique audio file in the table of detections. The audio recordings must be
        synchronized such that timestamps from each recording correspond to the exact same
        real-world time.

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
            detections: a dictionary of detections, with multi-index (file,start_time,end_time), and
                one column per class with 0/1 values for non-detection/detection The times in the
                index imply the same real world time across all files: eg 0 seconds assumes that the
                audio files all started at the same time, not on different dates/times

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
                time delays of a set of detections. [Default: 'gillette'] Options:
                    - 'gillette': linear closed-form algorithm of Gillette and Silverman 2008 [1]
                    - 'soundfinder': GPS location algorithm of Wilson et al. 2014 [2]

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

        # list of PositionEstimates for events that were successfully localized and passed filters
        localized_positions = [
            e
            for e in position_estimates
            if (e.location_estimate is None or e.residual_rms > residual_threshold)
        ]

        # PositionEstimates for events that we do not consider valid localizations (e.g. too few
        # receivers, too few receivers after applying cc_threshold or high residual in localization)
        unlocalized_positions = [
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
            return unlocalized_positions, localized_positions
        else:
            return unlocalized_positions

    def check_files_missing_coordinates(self, detections):
        """
        Check that all files in detections have coordinates in file_coords
        Returns:
            - a list of files that are in detections but not in file_coords
        """
        files_missing_coordinates = []
        files = list(detections.reset_index()["file"].unique())
        for file in files:
            if str(file) not in self.file_coords.index:
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
        # check that all files have coordinates in file_coords
        if len(self.check_files_missing_coordinates(detections)) > 0:
            raise UserWarning(
                "WARNING: Not all audio files have corresponding coordinates in self.file_coords."
                "Check file_coords.index contains each file in detections.index. "
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
        nearby_files_dict = self.make_nearby_files_dict(max_receiver_dist)

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
                    # first, use pre-created dictionary of nearby receivers for each audio file
                    close_receivers = nearby_files_dict[ref_file]
                    # then, subset files with detections to those that are nearby
                    close_det_files = [f for f in files_w_dets if f in close_receivers]

                    # if enough receivers, create a SpatialEvent using this set of receivers
                    # +1 to count the reference receiver
                    if len(close_det_files) + 1 >= min_n_receivers:
                        # SpatialEvent will include reference receiver + close recievers
                        receiver_files = [ref_file] + close_det_files

                        # retrieve positions of each receiver
                        receiver_locations = [
                            self.file_coords.loc[r] for r in receiver_files
                        ]

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

    def make_nearby_files_dict(self, r_max):
        """create dictinoary listing nearby files for each file

        pre-generate a dictionary listing all close files for each audio file
        dictionary will have a key for each audio file, and value listing all other receivers
        within r_max of that receiver

        eg {ARU_0.mp3: [ARU_1.mp3, ARU_2.mp3...], ARU_1... }

        Note: could manually create this dictionary to only list _simulataneous_ nearby
        files if the detection dataframe contains files from different times

        The returned dictionary is used in create_candidate_events as a look-up table for
            recordings nearby a detection in any given file

        Args:
            r_max: maximum distance from each recorder in which to include other
                recorders in the list of 'nearby recorders', in meters

        Returns:
            dictionary with keys for each file and values = list of nearby recordings
        """
        aru_files = self.file_coords.index.values
        nearby_files_dict = dict()
        for aru in aru_files:  # make an entry in the dictionary for each file
            reference_receiver = self.file_coords.loc[aru]  # location of receiver
            other_receivers = self.file_coords.drop([aru])
            distances = np.array(other_receivers) - np.array(reference_receiver)
            euclid_distances = [np.linalg.norm(d) for d in distances]

            # boolean mask for whether recorder is close enough
            mask = [r <= r_max for r in euclid_distances]
            nearby_files_dict[aru] = list(other_receivers[mask].index)

        return nearby_files_dict
