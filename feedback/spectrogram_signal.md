# Docstring Review: spectrogram.py and signal_processing.py

## Issues Fixed

### spectrogram.py

| Location | Issue | Fix Applied |
|---|---|---|
| `Spectrogram` class Attributes | Listed `window_type` (not in `__slots__`) and `overlap_samples` (wrong name; actual attribute is `hop_samples`); `fft_size` slot was undocumented | Removed `window_type`, renamed `overlap_samples` → `hop_samples`, added `fft_size`; updated type annotations from `list` → `np.ndarray` |
| `Spectrogram.pcen` | `**kwargs` and following text on same line; Returns description used informal phrasing | Separated `**kwargs` description; standardised Returns type annotation |
| `Spectrogram.window_length_seconds` | Trailing colon at end of one-line docstring | Removed trailing colon |
| `Spectrogram.min_max_scale` | "using in_range as minimum and maximum" — misleading; this method uses the data's actual min/max, not an explicit `in_range` parameter | Changed to "using the minimum and maximum values of the spectrogram" |
| `Spectrogram.linear_scale` | Extra blank line; `(low,high)` without spaces | Removed blank line; added spaces |
| `Spectrogram.net_amplitude` | `reject band:` in Args (wrong name; parameter is `reject_bands`); Returns used non-standard `return:` format | Fixed parameter name; converted to standard `Returns:` section with type annotation |
| `Spectrogram.bandpass` | "crops the 2-d array of the **spectrograms**" (plural) | Changed to "spectrogram" (singular) |
| `MelSpectrogram.from_audio` | `window_type="hann"` listed in Args but **not in the function signature**; `overlap_samples=None,` had trailing comma; `n_mels` default documented as 128 but actual default is 64; `power` documented but not in signature (passes via `**kwargs` without mention) | Removed `window_type`; fixed trailing comma; corrected `n_mels` default to 64; removed standalone `power` entry; added `**kwargs` documentation |
| `plot_spectrograms` | `frequency_range` parameter present in signature but **absent from Args section** | Added `frequency_range` to Args |
| `torch_to_dB` | **No docstring** | Added full docstring with Args and Returns |
| `numpy_to_dB` | **No docstring** | Added full docstring with Args and Returns |

### signal_processing.py

| Location | Issue | Fix Applied |
|---|---|---|
| `frequency2scale` | `pywt.ctw()` — typo (should be `pywt.cwt()`); `freuquency_hz` — typo | Fixed to `pywt.cwt()`; fixed to `frequency_hz` |
| `cwt_peaks` | `plot` parameter in signature but **absent from Args section** | Added `plot` parameter to Args |
| `find_accel_sequences` | "deteting" typo; "criterea" × 3 (should be "criteria"); `points_range` default in docstring listed as `(9, 100)` but actual default is `(5, 100)`; Args used old-style `param=default:` format | Fixed typos; corrected default value; normalised Args format to `param: description [default: value]` |
| `detect_peak_sequence_cwt` | "accellerating" typo; "threhsold" typo; `points_range` default listed as `(9, 100)` (actual is `(9, 100)` here — matches); Args used old-style `param=default:` format; Returns had no type annotation | Fixed typos; normalised Args format; added `pd.DataFrame` type to Returns |
| `thresholded_event_durations` | Returns: "duration (# samples/sr)" — ambiguous unit description referencing `sr` without definition | Clarified to "in seconds if sample_rate is in Hz" |
| `tdoa` | "samping" typo in Returns section | Fixed to "sampling" |

---

## API Design Suggestions

### spectrogram.py

1. **`Spectrogram.__init__` has no docstring.** The constructor accepts `spectrogram`, `power_spectrogram`, and `stft` as mutually exclusive inputs, which is a non-obvious pattern. A docstring on `__init__` explaining the three alternative input modes and what validation is performed would greatly help users who construct `Spectrogram` objects directly.

2. **Constructor argument naming — `stft` vs `magnitude`.** The class docstring (Properties section) refers to `self.magnitude`, while the constructor parameter is called `stft`. Meanwhile, the `magnitude` property returns `sqrt(power_spectrogram)`, not the raw complex STFT. The naming is inconsistent: `stft` implies complex-valued, but `magnitude` implies real-valued. Consider renaming the constructor parameter `stft` to `complex_stft` or keeping a note in the docstring.

3. **`amplitude()` raises `AttributeError` — undocumented removal.** The method has only `"""removed in favor of .rms property"""` as its docstring, but this is a deprecated public API. The docstring should at minimum document the deprecation and the replacement, e.g. with a `.. deprecated::` Sphinx note or a `DeprecationWarning`.

4. **`limit_range` shadows Python built-ins.** Parameters are named `min` and `max`, which shadow Python's built-in `min()` and `max()` functions. Prefer `min_db` / `max_db` (or `clip_min` / `clip_max`) to avoid confusion and potential bugs in subclass overrides.

5. **`to_image` `range` parameter shadows Python built-in.** Same issue: `range` shadows Python's `range()`. A name like `value_range` or `dB_range` would be less surprising.

6. **`plot` and `MelSpectrogram.plot` are siblings but have different signatures.** `Spectrogram.plot` accepts `cmap` and `dB` parameters; `MelSpectrogram.plot` does not (it always uses `'Greys'` and dB). For a consistent interface either both should accept `cmap`/`dB` or the base-class default should be documented as overrideable.

7. **`_spawn` is undocumented for its `spectrogram`/`stft` special-casing.** The method checks `if "spectrogram" not in slots` and injects `spectrogram=None`, but this is not documented. Users of subclasses who call `_spawn` could be surprised.

8. **`plot_spectrograms` / `plot_spectrograms_from_audio` — `n_col` not validated.** If `len(specs) == 1` and `n_col > 1`, the `axs[r, c]` indexing raises an `IndexError`. A validation or safe index into `axs` would be more robust. This is not a docstring issue but worth noting.

### signal_processing.py

9. **`cwt_peaks` `plot` parameter is undocumented side-effect behaviour.** The function both *returns values* and *optionally shows plots as a side effect*. Consider splitting into a pure computational function and a separate plotting helper, or at minimum clearly flagging the side-effect in the summary line.

10. **`find_accel_sequences` returns `(sequences_t, sequences_y)` — order is surprising.** Elsewhere in the codebase (e.g. `cwt_peaks`) the convention is `(times, values)`. The return tuple `(sequences_t, sequences_y)` is consistent with that. However, `detect_peak_sequence_cwt` internally destructures as `seq_t, seq_y = find_accel_sequences(...)`, which matches. The docstring and naming are fine but worth flagging that the y-values here are *inter-beat intervals* (forward differences of `t`), not amplitudes — this could be made clearer in the Returns description.

11. **`gcc` and `tdoa` — `frequency_range` is asymmetrically applied.** In the `gcc` implementation, only `X` (not `Y`) is zeroed outside the `frequency_range`. The docstring says "frequencies to keep in the GCC" without explaining this asymmetry. The asymmetry is intentional (attenuate signal only, not reference), but it should be documented.

12. **`tdoa` `return_max` returns inconsistent types.** When `return_max=False` it returns a scalar float; when `return_max=True` it returns a `(float, float)` tuple. The Returns section documents this, but a more Pythonic approach would be a dedicated `tdoa_with_confidence()` function or a named tuple / dataclass so callers don't need to branch on the return type.

13. **`detect_peak_sequence_cwt` `points_range` default inconsistency.** The function signature has `points_range=(9, 100)` while `find_accel_sequences` has `points_range=(5, 100)`. If `detect_peak_sequence_cwt` always passes `points_range` through to `find_accel_sequences`, the two defaults should be the same (or `detect_peak_sequence_cwt` should document why it uses a stricter default).
