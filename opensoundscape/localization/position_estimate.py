import pandas as pd
import numpy as np


class PositionEstimate:
    """class created by localization algorithms to store estimated location of a sound source

    Information about the sound event:
    - location_estimate: 2 or 3 element array of floats, estimated location of the sound source
    - class_name: string, class name of the sound source
    - start_timestamp: timestamp of the start of the event
    - duration: duration of the event in seconds


    Also contains information about the receivers used for localization, and intermediate outputs
    of the localization algorithm:
    - receiver_files: list of file paths to audio files used for localization
    - receiver_start_time_offsets: list of floats, time from start of audio to start of event
        for each receiver
    - receiver_locations: list of receiver locations
    - tdoas: list of time differences of arrival computed with cross correlation
    - cc_maxs: list of cross correlation maxima

    Args:
        location estimate: 2 or 3 element array of floats, estimated location of the sound source
        see SpatialEvent() for other args
    """

    def __init__(
        self,
        location_estimate,
        class_name=None,
        receiver_files=None,
        receiver_locations=None,
        tdoas=None,
        cc_maxs=None,
        start_timestamp=None,
        receiver_start_time_offsets=None,
        duration=None,
        distance_residuals=None,
    ):
        self.location_estimate = location_estimate
        self.class_name = class_name
        self.receiver_files = receiver_files
        self.receiver_locations = receiver_locations
        self.tdoas = tdoas
        self.cc_maxs = cc_maxs
        self.start_timestamp = start_timestamp
        self.receiver_start_time_offsets = receiver_start_time_offsets
        self.duration = duration
        self.distance_residuals = distance_residuals

    def __repr__(self):
        return f"PositionEstimate({self.location_estimate})"

    @property
    def residual_rms(self):
        # Calculate root mean square of distance residuals and store as attribute
        return np.sqrt(np.mean(self.distance_residuals**2))

    @classmethod
    def from_dict(cls, dictionary):
        """Recover PositionEstimate from dictionary, eg loaded from json"""
        array_keys = (
            "receiver_files",
            "receiver_locations",
            "tdoas",
            "cc_maxs",
            "location_estimate",
            "distance_residuals",
        )
        for key, val in dictionary.items():
            if key == "start_timestamp":
                dictionary[key] = pd.Timestamp(val)
            if key in array_keys:
                dictionary[key] = np.array(val)
        return cls(**dictionary)

    def to_dict(self):
        """PositionEstimate to json-able dictionary"""
        d = self.__dict__.copy()
        for key in d.keys():
            if isinstance(d[key], np.ndarray):
                d[key] = d[key].tolist()
            if isinstance(d[key], pd.Timestamp):
                d[key] = str(d[key])
        return d

    def load_aligned_audio_segments(self, start_offset=0, end_offset=0):
        """Load audio segments from each receiver at start times offset by tdoas

        This is useful for checking for correct alignment: the sound source should be aligned across
        all receivers if the tdoas (time differences of arrival) are correct.

        Note: requires self.receiver_start_time_offsets to determine starting position for each file

        Args:
            start_offset: float, amount of time before start of event to include in audio segment
            end_offset: float, amount of time after end of event to include in audio segment

        Returns:
            list of Audio objects, one for each receiver

        Example:

        ```python
        low_f = 1000 # Hz
        high_f = 5000 # Hz
        audio_segments = example.load_aligned_audio_segments()
        specs = [Spectrogram.from_audio(a).bandpass(low_f,high_f) for a in audio_segments]
        plt.pcolormesh(np.vstack([s.spectrogram for s in specs]),cmap='Greys')
        ```
        """
        from opensoundscape.audio import Audio

        all_audio = []
        for i, audio_path in enumerate(self.receiver_files):
            start = self.receiver_start_time_offsets[i] + self.tdoas[i] - start_offset
            all_audio.append(
                Audio.from_file(
                    audio_path,
                    offset=start,
                    duration=self.duration + start_offset + end_offset,
                    out_of_bounds_mode="warn",
                )
            )

        return all_audio


def positions_to_df(list_of_events):
    """convert a list of PositionEstimate objects to pd DataFrame

    Args:
        list_of_events: list of PositionEstimate objects

    Returns:
        pd.DataFrame with columns for each attribute of a PositionEstimate object

    See also: df_to_events, PositionEstimate.from_dict, PositionEstimate.to_dict
    """
    return pd.DataFrame([e.__dict__ for e in list_of_events])


def df_to_positions(df):
    """convert a pd DataFrame to list of PositionEstimate objects

    works best if df was created using `events_to_df`

    Args:
        df: pd.DataFrame with columns for each attribute of a PositionEstimate object

    Returns:
        list of PositionEstimate objects

    See also: events_to_df, PositionEstimate.from_dict, PositionEstimate.to_dict
    """
    return [PositionEstimate.from_dict(df.loc[i].to_dict()) for i in df.index]
