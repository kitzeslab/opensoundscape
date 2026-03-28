# API Design Suggestions and Issues: `opensoundscape/audio.py`

## Docstring Issues Found and Fixed

The following issues were identified and corrected:

| Location | Issue | Fix Applied |
|----------|-------|-------------|
| `Audio.silence` | Spurious `"` in opening triple-quote (`""" "Create...`), no Returns section | Fixed |
| `Audio.noise` | Spurious `"` in opening triple-quote; typo "implementatino"; missing `dBFS` param in Args; `[default: 'white']` was misindented | Fixed |
| `Audio.from_file` | "tinytag" (wrong library, uses aru_metadata_parser); "method used to resample_type" (nonsensical); "not full contained" -> "not fully contained" | Fixed |
| `Audio.trim_samples` | Typo "exlusive" -> "exclusive" | Fixed |
| `Audio.trim_with_timestamps` | Parameter description said "end_datetime" instead of "end_timestamp"; `out_of_bounds_mode` default shown as "ignore" when signature default is "warn" | Fixed |
| `Audio.loop` | Typo "occurences" -> "occurrences" | Fixed |
| `Audio.extend_by` | `duration` described as "the final duration" when it is actually the amount of silence to add | Fixed |
| `Audio.bandpass` | "solfiltfilt" -> "sosfiltfilt"; missing Returns section | Fixed |
| `Audio.lowpass` | "solfiltfilt" -> "sosfiltfilt"; "cuttof_f" -> "cutoff_f"; missing Returns section | Fixed |
| `Audio.highpass` | Missing Returns section | Fixed |
| `Audio.spectrum` | Listed `self` in Args section (never appropriate for instance methods) | Fixed |
| `Audio.normalize` | Lowercase `args:` and `returns:` headings (should be `Args:` / `Returns:`) | Fixed |
| `Audio.save` | "soundfinder" -> "soundfile" in both docstring and error message (code bug also fixed) | Fixed |
| `Audio.split` | "even-lengthed" -> "even-length"; **kwargs Args description was malformed | Fixed |
| `Audio.apply` | `return_df` listed in Args but is not a parameter in the function signature | Fixed |
| `Audio.split_and_save` | `final_clip` default shown as `[default: None]` when signature default is `"extend"`; misindented `[default: None]` line | Fixed |
| `Audio._get_sample_index` | "The time to multiply with the sample_rate" was inaccurate; Returns said "rounded sample" | Fixed |
| `load_channels_as_audio` | `args:` / `returns:` lowercase; Args just said "see Audio.from_file()" without listing params | Fixed |
| `parse_opso_metadata` | Typo "opensoundcsape" -> "opensoundscape" | Fixed |
| `generate_opso_metadata_str` | Typo "fundtion" -> "function" | Fixed |
| `write_metadata` | "Audio.wave documentation" -> "Audio.save documentation" (wrong method name) | Fixed |
| `lowpass_filter` | Args listed `low_f` and `high_f` which do not exist in the signature; should be `cutoff_f`; "-3db" -> "-3 dB" | Fixed |
| `bandpass_filter` | Described both `low_f` and `high_f` as "-3db point for **highpass** filter" (wrong filter type); "-3db" -> "-3 dB" | Fixed |
| `clipping_detector` | `threshold=0.6:` put the default in the argument name, not in the description | Fixed |
| `estimate_delay` | Description referred to "audio" instead of "primary_audio"; unclosed parenthesis in the NOTE | Fixed |
| `_audio_from_file_handler` | "tinytag" -> "aru_metadata_parser"; "resample_type: method used to resample_type"; "not full contained"; typo "provied"; missing `cls` in Args | Fixed |
| `MultiChannelAudio.silence` | Spurious `"` in opening triple-quote; missing `channels` param in Args; missing Returns | Fixed |
| `MultiChannelAudio.noise` | Entire docstring missing | Added |
| `MultiChannelAudio.from_file` | Copied from `Audio.from_file` verbatim, incorrectly saying files "are mixed down to mono" and directing users to `load_channels_as_audio()`; default for `out_of_bounds_mode` shown as 'warn' when signature has 'ignore' | Fixed |
| `MultiChannelAudio.extend_to` | Returns said "Audio object" instead of "MultiChannelAudio object" | Fixed |
| `MultiChannelAudio.spectrum` | Listed `self` in Args; description said "an Audio object" instead of "a MultiChannelAudio object" | Fixed |
| `MultiChannelAudio.split_and_save` | Same `final_clip` default/indentation issue as `Audio.split_and_save` | Fixed |
| `MultiChannelAudio.apply_channel_gain` | Returns said "Audio object" instead of "MultiChannelAudio object" | Fixed |
| `MultiChannelAudio.n_channels` | Missing docstring | Added |
| `MultiChannelAudio.duration` | Missing docstring | Added |
| `MultiChannelAudio.to_mono` | Missing docstring | Added |
| `MultiChannelAudio.to_channels` | Missing docstring | Added |
| `parse_metadata` (warning string) | Runtime warning said "opensoundcape" (code bug, not docstring) | Fixed |

---

## API Design Suggestions

### 1. Inconsistent `out_of_bounds_mode` defaults across methods
`Audio.from_file` defaults to `"warn"`, `Audio.trim` / `Audio.trim_samples` default to `"ignore"`,
and `MultiChannelAudio.from_file` defaults to `"ignore"`. A single consistent default across all
methods would reduce user confusion.

### 2. `Audio.silence` and `MultiChannelAudio.silence` have different signatures
`Audio.silence(duration, sample_rate)` vs `MultiChannelAudio.silence(duration, sample_rate, channels)`.
Since `MultiChannelAudio` subclasses `Audio`, polymorphic callers cannot use them interchangeably.
Consider giving `Audio.silence` a `channels=1` parameter, routing to `MultiChannelAudio.silence`
when `channels > 1`, or at minimum documenting this difference clearly.

### 3. `split` returns a tuple `(clips, clip_df)` — confusing API
`split()` returns `(list_of_Audio, DataFrame)`. Most callers only want the clips. A common footgun
is writing `clips = audio.split(...)` and getting a tuple. Consider returning only the list and
providing a separate `split_with_times()` variant, or making the return consistent with how
`split_and_save` works (which returns only a DataFrame).

### 4. `apply` does not document (nor implement) `return_df`
The old docstring mentioned a `return_df` parameter that was never in the signature. This suggests
a feature was either removed or never implemented. If desired, the parameter should be added to
the signature; otherwise, remove the reference (done in this PR).

### 5. `MultiChannelAudio.split_and_save` always raises `NotImplementedError`
The method exists (with full docstring) but the body is just `raise NotImplementedError`. Either
implement it or replace with a proper `raise NotImplementedError("Not yet implemented for MultiChannelAudio")`.

### 6. `extend_to` vs `extend_by` asymmetry
`extend_to(duration)` extends to a *minimum* total duration; `extend_by(duration)` extends by an
*exact* amount. However, `extend_to` silently returns `self` unchanged if the audio is already
long enough, while `extend_by` asserts `duration >= 0`. Consider returning a copy (via `_spawn()`)
rather than `self` in `extend_to` for consistency with the rest of the out-of-place API.

### 7. `pad_to` returns `self` (not a copy) when no padding is needed
`pad_to` has `return self` when `duration <= self.duration`, while all other methods use
`return self._spawn()`. This breaks the immutability contract and can cause subtle bugs where the
caller's reference shares state with the original object.

### 8. `MultiChannelAudio.extend_to` metadata duration calculation bug
In `MultiChannelAudio.extend_to`, `len(new_samples)` returns the number of channels (first
dimension), not the number of samples per channel. It should be `new_samples.shape[1]` to match
the `Audio.extend_to` behavior. (This is a correctness bug, not a docstring issue.)

### 9. `concat` discards metadata without warning
The `concat` docstring notes it "discards metadata" but does not warn the user at runtime. Since
metadata (e.g., `recording_start_time`) is often important, consider either preserving/merging
metadata or emitting a warning when metadata is discarded.

### 10. `estimate_delay` example in docstring uses `{...}` code block syntax
The code examples in `Audio.from_file` wrap the example in `{...}` which looks like a dictionary
literal and cannot be copy-pasted directly. The `_audio_from_file_handler` version has already
been corrected to remove the braces, but `Audio.from_file` still has them (minor inconsistency).

### 11. `dBFS` property naming
The property `Audio.dBFS` calculates RMS-based dBFS scaled for a sine wave (`rms * sqrt(2)`).
This is an unusual definition. The docstring should clarify that this measures RMS level
referenced to a full-scale *sine wave*, and how this differs from peak dBFS.

### 12. `__add__` operator inconsistency
`audio + other_audio` → concatenation, `audio + 3` → gain (dB adjustment). Mixing these two
semantically very different behaviors under `+` is surprising. The gain behavior via `+` is
non-standard (normally `+` would mean element-wise addition of sample arrays). Consider
reserving `+` only for concatenation and using explicit `apply_gain()` for gain, or at minimum
document this clearly in the class-level docstring.
