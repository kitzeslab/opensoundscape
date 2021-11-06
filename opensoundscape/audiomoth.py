"""Utilities specifically for audio files recoreded by AudioMoths"""
import pytz
import datetime
from opensoundscape.helpers import hex_to_time


def audiomoth_start_time(file, filename_timezone="UTC", to_utc=False):
    """parse audiomoth file name into a time stamp

    AudioMoths create their file name based on the time that recording starts.
    This function parses the name into a timestamp. Older AudioMoth firmwares
    used a hexidecimal unix time format, while newer firmwares use a
    human-readable naming convention. This function handles both conventions.

    Args:
        file: (str) path or file name from AudioMoth recording
        filename_timezone: (str) name of a pytz time zone (for options see
            pytz.all_timezones). This is the time zone that the AudioMoth
            uses to record its name, not the time zone local to the recording
            site. Usually, this is 'UTC' because the AudioMoth records file
            names in UTC.
        to_utc: if True, converts timestamps to UTC localized time stamp.
            Otherwise, will return timestamp localized to `timezone` argument
            [default: False]

    Returns:
        localized datetime object
        - if to_utc=True, datetime is always "localized" to UTC
    """
    name = Path(file).stem
    if len(name) == 8:
        # HEX filename convention (old firmware)
        dt = hex_to_time(Path(file).stem)
    elif len(name) == 15:
        # human-readable format (newer firmware)
        dt = datetime.datetime.strptime(name, "%Y%m%d_%H%M%S")
    else:
        raise ValueError(f"file had unsupported name format: {name}")

    # convert the naive datetime into a localized datetime based on the
    # timezone provided by the user. (This is the time zone that the AudioMoth
    # uses to record its name, not the time zone local to the recording site.)
    localized_dt = pytz.timezone(timezone).localize(dt)

    if to_utc:
        return localized_dt.astimezone(pytz.utc)
    else:
        return localized_dt


def parse_audiomoth_metadata(metadata):
    """parse a dictionary of AudioMoth .wav file metadata

    -parses the comment field
    -adds keys for gain_setting, battery_state, recording_start_time

    Tested for AudioMoth firmware versions:
        1.5.0
        ...?

    Args:
        metadata: dictionary with audiomoth metadata

    Returns:
        metadata dictionary with added keys and values
    """
    import datetime
    import pytz

    comment = metadata["comment"]
    timezone = pytz.timezone(comment.split("(")[1].split(")")[0])
    datetime_str = comment.split("Recorded at ")[1][:19]
    metadata["recording_start_time"] = timezone.localize(
        datetime.datetime.strptime(datetime_str, "%H:%M:%S %d/%m/%Y")
    )
    metadata["gain_setting"] = int(comment.split("gain setting ")[1][:1])
    metadata["battery_state"] = float(comment.split("battery state was ")[1][:3])
    metadata["audiomoth_id"] = metadata["artist"].split(" ")[1]
    return metadata


def parse_audiomoth_metadata_from_path(file_path):
    from tinytag import TinyTag

    metadata = TinyTag.get(file_path).as_dict()
    return parse_audiomoth_metadata(metadata)
