Select recordings by date and time
==================================

These helpers filter by recording start, preserving path order and duplicates.
They read header metadata without decoding audio samples. Both boundaries are
inclusive; recordings merely overlapping a window are not selected.

.. code-block:: python

    from datetime import date, time
    from pathlib import Path
    from opensoundscape.utils import (
        select_files_in_date_range, select_files_in_time_range,
    )

    files = sorted(Path("recordings").glob("*.WAV"))

    selected = select_files_in_date_range(
        files, date(2024, 1, 1), date(2024, 1, 31), "America/New_York"
    )
    overnight = select_files_in_time_range(
        selected, time(22), time(2), "America/New_York"
    )

Filename fallback is opt-in and uses aru-metadata-parser's recorder patterns
and best-effort generic parsing. Metadata always takes precedence.

.. code-block:: python

    selected = select_files_in_date_range(
        files, date(2024, 1, 1), date(2024, 1, 31), "America/New_York",
        parse_timestamp_from_filename=True,
        file_timezone="UTC",
        raise_exceptions=False,
    )

``file_timezone`` describes the recorder clock, whereas ``target_timezone``
(the fourth argument above) controls selection. For example, midnight UTC on
January 2 is January 1 in New York. An embedded offset must agree with
``file_timezone``; conflicting offsets are timestamp errors and are never overridden.

Files with timestamp errors are skipped by default. Set
``raise_exceptions=True`` to raise ``GetRecordingStartTimeError`` with the
original exception as its cause. Errors include missing timestamps, conflicting
timezone offsets, naive timestamps without a supplied zone, and ambiguous or
nonexistent local times during daylight-saving transitions. Invalid date/time
bounds and unrecognized timezone names raise before processing files. File
modification times and the computer's local timezone are never used.

Date bounds must be ``datetime.date`` values, not datetimes. Time bounds must be
``datetime.time`` values without timezones. An earlier ending time wraps midnight;
equal time bounds match exactly, including microseconds.
