"""preprocess.py: utilities for augmentation and preprocessing pipelines"""

def random_audio_trim(self, audio, duration,extend_short_clips=False):
    """randomly select a subsegment of Audio of fixed length

    randomly chooses a time segment of the entire Audio object to cut out,
    from the set of all possible start times that allow a complete extraction

    Args:
        Audio: input Audio object
        length: duration in seconds of the trimmed Audio output

    Returns:
        Audio object trimmed from original
    """
    input_duration = len(audio.samples) / audio.sample_rate
    if duration > input_duration:
        if not extend_short_clips:
            raise ValueError(
                f"the length of the original file ({input_duration} sec) was less than the length to extract ({duration} sec). To extend short clips, use extend_short_clips=True"
            )
        else:
            return audio.extend(duration)
    extra_time = input_duration - duration
    start_time = np.random.uniform() * extra_time
    return audio.trim(start_time, start_time + duration)
