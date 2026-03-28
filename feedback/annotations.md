# Feedback: `opensoundscape/annotations.py`

## Issues Fixed in This PR

- `to_raven_files`: "save annotations to **a** Raven-compatible tab-separated text files" → removed spurious "a"
- `to_crowsetta` (x2): "creates **on** Annotation/Sequence object" → "creates **one** Annotation/Sequence object"
- `to_crowsetta` `ignore_sequence_id`: description incorrectly referenced `annotation_id` instead of `sequence_id`
- `bandpass`: "high **frequench**" → "high **frequency**"
- `unique_labels`: "null/Falsy labels" — `dropna()` only drops NaN, not all falsy values (empty string, 0, etc.); corrected to "null/NaN labels"
- `labels_on_index` / `clip_labels` (x2 each): "**criterea**" → "**criteria**"; "ths parameter" → "this parameter"
- `labels_on_index`: module path typo "**opensoundscap**.utils.make_clip_df" → "**opensoundscape**.utils.make_clip_df" (appeared twice)
- `find_overlapping_idxs_in_clip_df`: `Returns:` section was missing proper indentation and wrapping; also missing blank line before `Args:` section; fixed formatting
- `train_test_split`: missing `Returns:` section — added
- `integer_to_multi_hot`: `sparse` parameter was missing from `Args:` — added
- `concat`: missing `Args:` and `Returns:` sections — added
- `from_categorical_labels_df`: missing `Returns:` section — added; first-line description was a run-on sentence
- `from_multihot_df`: missing `Returns:` section — added
- `multihot_df_sparse` property: "**parse** dataframe" → "**sparse** dataframe"
- `multihot_array` / `multihot_df`: very brief docstrings with no `Args:` or `Returns:` — added
- `labels_at_index` / `multihot_labels_at_index`: one-liner with no `Args:` or `Returns:` — expanded
- `categorical_to_integer_labels` / `integer_to_categorical_labels`: bare one-liners; added `Args:` and `Returns:` sections
- `from_crowsetta_bbox` / `from_crowsetta_seq`: "this classmethod is used by from_crowsetta()" appeared after `Returns:`, which is non-standard; moved to a `Note:` section at the top of the docstring
- `from_csv`: "load csv from path and **creates**" → "load CSV from path and **create**" (imperative mood; consistent capitalisation)
- `CategoricalLabels.__init__`: class overview (ClassMethods/Methods/Properties) belonged in a class-level docstring, not in `__init__`; moved there and cleaned up `__init__` docstring to focus on parameters

---

## API Design Suggestions

### Naming inconsistencies

1. **`to_raven_files` vs `from_raven_files`**: the `from_raven_files` classmethod accepts a `column_mapping_dict` parameter but the code silently overwrites it with hardcoded defaults before applying `column_mapping_dict.update(column_mapping_dict or {})` — this call updates the dict with itself, so any user-supplied mapping is **never applied**. This is a latent bug: the parameter is documented but non-functional.

2. **`annotation_files` / `audio_files` vs `.df['annotation_file']` / `.df['audio_file']`**: two parallel representations of file lists (instance attributes vs DataFrame columns) can fall out of sync. There is no enforcement that `audio_files` matches `df['audio_file'].unique()`. This frequently surprises users (e.g., `to_raven_files` uses instance attributes, not DataFrame column values).

3. **`to_csv` uses "Effects:" section, `to_raven_files` uses "Outcomes:" section**: inconsistent side-effect documentation across the class. Standardise on one (e.g., always `Note:` or always `Returns:` with a "None" description and a note about side effects).

4. **`from_crowsetta_bbox` and `from_crowsetta_seq`** are public classmethods but are internal helpers. Consider renaming to `_from_crowsetta_bbox` / `_from_crowsetta_seq` (leading underscore) to signal they are implementation details.

5. **`global_multi_hot_labels`**: returns a plain `list` of `int` rather than a numpy array or DataFrame. Inconsistent with the rest of the label-generation methods which return DataFrames or arrays.

6. **`labels_on_index`**: the `return_type='integers'` option returns a `(df, classes)` tuple while `return_type='multihot'` returns just a DataFrame. This inconsistency (tuple vs scalar) is error-prone. Consider always returning a consistent structure (e.g., always return a named tuple or always return the DataFrame with `classes` accessible via a separate attribute/method).

7. **`CategoricalLabels`**: `start_times` and `end_times` constructor arguments are poorly named — they refer to clip-level temporal bounds, but for many use-cases the concept is "sample" (file + clip). Using `clip_start_times` / `clip_end_times` would be clearer.

8. **`diff()`**: the function exists, is documented with a docstring, but immediately raises `NotImplementedError`. Either implement it or remove it from the public API to avoid confusion.

9. **`unique()` module-level helper**: this is a useful utility but is not exported or mentioned in `__all__`. Since it is not prefixed with `_`, it is technically public but undiscoverable.

10. **`CategoricalLabels.multihot_df(sparse=True)` default**: most pandas users expect DataFrames to be dense by default. Defaulting to `sparse=True` is unusual and may surprise users; consider changing the default to `False` and documenting the sparse variant as an opt-in.
