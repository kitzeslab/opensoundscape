from opensoundscape.audio import Audio
from pathlib import Path
from opensoundscape.utils import make_clip_df

SAMPLE_DATA_ROOT = (Path(__file__).parent) / "sample_data"
SAMPLE_AUDIO_PATH = SAMPLE_DATA_ROOT / "birds_10s.wav"

import sys


class SampleData(sys.__class__):  # sys.__class__ is <class 'module'>

    @property
    def birds(self):
        """load a 10s clip of birdsong from Pennsylvania"""
        return Audio.from_file(SAMPLE_AUDIO_PATH)

    @property
    def birds_path(self):
        """returns a path to a sample 10s audio clip included in Opensoundscape"""
        return SAMPLE_AUDIO_PATH

    def clip_df(self, clip_duration=5, **kwargs):
        return make_clip_df([SAMPLE_AUDIO_PATH], clip_duration=clip_duration, **kwargs)


# "accepted" hack for replacing this module with the SampleData class
# this allows properties to be accessed as if they were module attributes
sys.modules[__name__].__class__ = SampleData  # change module class into This
