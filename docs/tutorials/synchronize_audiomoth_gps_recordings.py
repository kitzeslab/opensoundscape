"""
sync all audio from an entire dataset by using the PPS data to generate audio files starting at a
known time and having the exact desired sampling rate creates a table with records of per-file
success or failure and saves each resampled, syncrhonized audio file
"""

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import concurrent.futures

from opensoundscape import Audio
from opensoundscape.localization.audiomoth_sync import (
    correct_sample_rate,
    associate_pps_samples_timestamps,
)

## parameters to modify ##

output_sr = 24000  # desired final sample rate for all resampled audio
# path containing sub-folders with audio and metadata files
audio_root = Path("/path/to/all/audio/folders")
# path to save synchronized resampled audio files to
output_folder = Path("/path/to/save")

# find all folders in the audio root, where folders contain .WAV and .CSV files
# assumes that each WAV file has a matching .CSV file of the same name
# change this line to correctly find your audio folders.
# In this case, we are looking for folders in the audio_root directory.
audio_folders = list(audio_root.glob("*"))
print(f"Found {len(audio_folders)} folders in audio_root.")

cpu_workers = 4  # number of parallel processes to use

# compatability with older GPS firmwares such as AudioMothGPSDeploy_1_0_8_Hardware_1_1
PPS_SUFFIX = ".CSV"  # use .PPS for older custom firmware, which saved files as .PPS
cpu_clock_counter_col = "TIMER_COUNT"  # use "COUNTER" for older firmware

# skip completed files or repeat?
# if True, if output file exists, skips
# set to False to overwrite outputs
skip_if_completed = True

# raise or catch & log errors?
raise_exceptions = False

## utilities (probably do not modify) ##


def sync_entire_folder(folder):
    """find all .WAV and corresponding .CSV files in a folder, synchronize, and save to output_folder"""
    audio_files = list(folder.glob("*.WAV"))
    success_record = []

    # make the directory for each sd card in the output folder
    out_sd = output_folder / folder.stem
    out_sd.mkdir(exist_ok=True, parents=True)
    print(f"There are {len(audio_files)} files in {folder}")

    for file in audio_files:
        # check if already completed
        out_filename = out_sd / file.name
        if out_filename.exists() and skip_if_completed:
            success_record.append(True)
            continue
        try:
            # Get the processed PPS DF
            pps_file = file.parent / str(file.stem + PPS_SUFFIX)
            assert pps_file.exists()

            processed_pps_df = associate_pps_samples_timestamps(
                pps_file, cpu_clock_counter_col=cpu_clock_counter_col
            )

            # Resample the audio
            resampled_audio = correct_sample_rate(
                Audio.from_file(file), processed_pps_df, desired_sr=output_sr
            )

            # save
            processed_pps_df.to_csv(out_sd / file.with_suffix(".csv").name)
            resampled_audio.save(out_filename)
            success_record.append(True)
        except Exception as exc:
            if raise_exceptions:
                raise exc
            print(f"Exception was raised while attempting to resample/sync {file}:")
            print(exc)
            success_record.append(False)

    return pd.DataFrame(
        zip(audio_files, success_record), columns=["audio_file", "success"]
    )


if __name__ == "__main__":
    """treat each folder as a separate task, parallelize each task with CPU workers"""
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as executor:
        sd_folders = tqdm(audio_folders)
        all_success_records = list(executor.map(sync_entire_folder, sd_folders))

        pd.concat(all_success_records).to_csv(
            output_folder / Path("sync_success_fail_record.csv")
        )
