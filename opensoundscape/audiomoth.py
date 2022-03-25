"""Utilities specifically for audio files recoreded by AudioMoths"""
import pytz
import datetime
from opensoundscape.helpers import hex_to_time
from pathlib import Path


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
        if filename_timezone != "UTC":
            raise ValueError('hexidecimal file names must have filename_timezone="UTC"')
        localized_dt = hex_to_time(Path(file).stem)  # returns UTC localized dt
    elif len(name) == 15:
        # human-readable format (newer firmware)
        dt = datetime.datetime.strptime(name, "%Y%m%d_%H%M%S")

        # convert the naive datetime into a localized datetime based on the
        # timezone provided by the user. (This is the time zone that the AudioMoth
        # uses to record its name, not the time zone local to the recording site.)
        localized_dt = pytz.timezone(filename_timezone).localize(dt)

    else:
        raise ValueError(f"file had unsupported name format: {name}")

    if to_utc:
        return localized_dt.astimezone(pytz.utc)
    else:
        return localized_dt


def parse_audiomoth_metadata(metadata):
    """parse a dictionary of AudioMoth .wav file metadata

    -parses the comment field
    -adds keys for gain_setting, battery_state, recording_start_time
    -if available (firmware >=1.4.0), addes temperature

    Notes on comment field:
    - Starting with Firmware 1.4.0, the audiomoth logs Temperature to the
      metadata (wav header) eg "and temperature was 11.2C."
    - At some point the firmware shifted from writing "gain setting 2" to
      "medium gain setting". Should handle both modes.

    Tested for AudioMoth firmware versions:
        1.5.0

    Args:
        metadata: dictionary with audiomoth metadata

    Returns:
        metadata dictionary with added keys and values
    """
    import datetime
    import pytz

    comment = metadata["comment"]

    # parse recording start time (can have timzeone info like "UTC-5")
    metadata["recording_start_time"] = _parse_audiomoth_comment_dt(comment)

    # gain setting can be written "medium gain" or "gain setting 2"
    try:
        metadata["gain_setting"] = int(comment.split("gain setting ")[1][:1])
    except ValueError:
        metadata["gain_setting"] = comment.split(" gain setting")[0].split(" ")[-1]
    # written "3.2V" or "less than 2.5V" (or? greater than 4.5V?)
    metadata["battery_state"] = _parse_audiomoth_battery_info(comment)
    metadata["audiomoth_id"] = metadata["artist"].split(" ")[1]
    if "temperature" in comment:
        metadata["temperature_C"] = float(
            comment.split("temperature was ")[1].split("C")[0]
        )

    return metadata


def parse_audiomoth_metadata_from_path(file_path):
    from tinytag import TinyTag

    metadata = TinyTag.get(file_path)

    if metadata is None:
        raise ValueError(f"{file_path} does not contain metadata")
    else:
        metadata = metadata.as_dict()
        artist = metadata["artist"]
        if not artist or (not "AudioMoth" in artist):
            raise ValueError(
                f"It looks like the file: {file_path} does not contain AudioMoth metadata."
            )
        else:
            return parse_audiomoth_metadata(metadata)


def _parse_audiomoth_comment_dt(comment):
    """parses start times as written in metadata Comment field of AudioMoths

    examples of Comment Field date-times:
    19:22:55 14/12/2020 (UTC-5)
    10:00:00 15/05/2021 (UTC)

    note that UTC-5 is not parseable by datetime, hence custom parsing
    also note day-month-year format of date

    Args:
        comment: the full comment string from an audiomoth metadata Comment field
    Returns:
        localized datetime object in timezone specified by original metadata
    """
    # extract relevant portion of comment
    dt_str = comment.split("Recorded at ")[1].split(" by ")[0]

    # handle formats like "UTC-5" or "UTC+0130"
    if "UTC-" in dt_str or "UTC+" in dt_str:
        marker = "UTC-" if "UTC-" in dt_str else "UTC+"
        dt_str_utc_offset = dt_str.split(marker)[1][:-1]
        if len(dt_str_utc_offset) <= 2:
            dt_str_tz_str = f"{marker}{int(dt_str_utc_offset):02n}00"
        else:
            dt_str_tz_str = f"{marker}{int(dt_str_utc_offset):04n}"

        dt_str = f"{dt_str.split(marker)[0]}{dt_str_tz_str})"
    else:  #
        dt_str = dt_str.replace("(UTC)", "(UTC-0000)")
    dt = datetime.datetime.strptime(
        dt_str, "%H:%M:%S %d/%m/%Y (%Z%z)"
    )  # .astimezone(final_tz)
    return dt


def _parse_audiomoth_battery_info(comment):
    """attempt to parse battery info from metadata comment

    examples:
    ...battery state was 4.7V.
    ...battery state was less than 2.5V
    ...battery state was 3.5V and temperature....

    Args:
        comment: the full comment string from an audiomoth metadata Comment field
    Returns:
        float of voltage or string describing voltage, eg "less than 2.5V"
    """
    battery_str = comment.split("battery state was ")[1].split("V")[0] + "V"
    if len(battery_str) == 4:
        return float(battery_str[:-1])
    else:
        return battery_str
