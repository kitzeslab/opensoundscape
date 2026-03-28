# OpenSoundscape API Feedback

This document captures API design feedback and recommendations identified during a thorough review of the OpenSoundscape codebase. It is intended to guide refinements toward a stable v1.0.0 release.

---

## Summary

OpenSoundscape has a well-structured API centered around the `CNN`/`SpectrogramClassifier` model class, a rich preprocessing pipeline, and strong integration with modern tooling (W&B, Lightning, hoplite). The suggestions below focus on consistency, discoverability, and long-term maintainability.

---

## 1. Naming Consistency

### 1.1 `CNN` vs `SpectrogramClassifier`
`CNN` is an alias for `SpectrogramClassifier`. The alias is convenient but may cause confusion:
- Users may not realize the two names refer to the same class.
- The name `CNN` is technically incorrect for non-CNN architectures (e.g., ViT or transformer-based).
- **Recommendation:** Deprecate `CNN` in favor of `SpectrogramClassifier` for v1.0.0, or document the alias clearly in the class docstring and migration guide.

### 1.2 `predict` vs `__call__`
The model exposes both `.predict()` (high-level, returns a DataFrame) and `.__call__()` (low-level, returns a dict of numpy arrays). The distinction is not immediately obvious.
- **Recommendation:** Rename `__call__` to something like `_run_inference()` or `forward_batch()` to clarify it is a low-level method, and reserve `__call__` for the high-level predict workflow.

### 1.3 `fit_with_trainer` vs `train`
`SpectrogramClassifier` uses `.train()` while `LightningSpectrogramModule` uses `.fit_with_trainer()`. The asymmetry is confusing.
- **Recommendation:** Use `.fit()` for the high-level training entry point across both classes, consistent with scikit-learn and Lightning conventions.

---

## 2. Argument Naming

### 2.1 `num_workers=0` vs `num_workers=1`
The convention of using `num_workers=0` to mean "current process" is PyTorch-specific and surprising to users coming from other frameworks.
- **Recommendation:** Add a note to all `num_workers` docstrings clarifying that `0` means single-process (not "1 worker"), which is the current behavior. This is already documented in some places but inconsistently.

### 2.2 `audio_root` parameter
The `audio_root` parameter is duplicated across many methods (`predict`, `embed`, `train`, `generate_cams`, etc.). It is always optional and prepended to file paths.
- **Recommendation:** Consider storing `audio_root` as an instance attribute (e.g., `model.audio_root`) that can be set once rather than passing it to every method call. A per-call override could still be supported.

### 2.3 `clip_overlap` vs `clip_overlap_fraction` vs `overlap_fraction`
Three related parameters control clip overlap. `overlap_fraction` is deprecated but still present in multiple method signatures.
- **Recommendation:** Remove `overlap_fraction` in v1.0.0. Standardize on `clip_overlap_fraction` (a fraction) and `clip_overlap` (absolute seconds) across all relevant APIs.

---

## 3. Return Value Consistency

### 3.1 `predict()` output
`.predict()` returns a `pd.DataFrame` with `(file, start_time, end_time)` multi-index and one column per class. This is clear and consistent.
- No change needed.

### 3.2 `embed()` output
`.embed()` can return either a `pd.DataFrame` (if `return_dfs=True`) or a `np.ndarray` (if `return_dfs=False` or `avgpool=False`). The type depends on multiple parameters.
- **Recommendation:** Always return a `pd.DataFrame` when possible; use `return_dfs=False` only as an explicit override. Consider dropping the `avgpool` parameter from `embed()` and making it a separate lower-level method.

### 3.3 `generate_samples()` output
`.generate_samples()` returns a list of `AudioSample` objects, but can also return `(samples, invalid_samples)` if `return_invalid_samples=True`.
- **Recommendation:** For methods that optionally return extra info via a flag, prefer a consistent pattern (e.g., always return a named tuple or dataclass, or provide a separate method for invalid sample reporting).

---

## 4. Preprocessing Pipeline

### 4.1 Action Pipeline Mutability
`Preprocessor.pipeline` is a mutable `pd.Series`. Users can accidentally mutate it in unexpected ways.
- **Recommendation:** Provide convenience methods for common mutations (`add_action`, `remove_action`, `replace_action`) and consider documenting the recommended workflow for customizing the pipeline.

### 4.2 `forward()` Signature
`BasePreprocessor.forward()` has a `trace` and a `profile` argument that store output in the returned `AudioSample`. The behavior is useful but undiscoverable.
- **Recommendation:** Add example usage to the `forward()` docstring showing how to access `sample.trace` and `sample.runtime`.

---

## 5. Model Save/Load

### 5.1 `pickle=True` vs `pickle=False`
The `pickle` parameter of `.save()` controls two very different save formats. The default is `pickle=False`, which is better for sharing.
- **Recommendation:** Rename `pickle=True` to something more descriptive like `full_state=True` or `resume_training=True` to convey its intended use.

### 5.2 `unpickle=True` in `.load()`
Similarly, `CNN.load(unpickle=True)` is confusing since `unpickle` is the _default_ and means "allow loading pickled files."
- **Recommendation:** Rename to `allow_pickle=True` to align with numpy convention and improve clarity.

---

## 6. Device Handling

`model.device` is a property that accepts a string or `torch.device` and automatically converts. This is user-friendly.
- **No change needed.** However, the current behavior of defaulting to `cuda:0` when multiple GPUs are available may be surprising—consider a note documenting this.

---

## 7. Localization Module

### 7.1 Consistent "receiver" spelling
The localization module had widespread use of "reciever" (misspelled). This has been corrected in this PR.

### 7.2 `SpatialEvent` API
`SpatialEvent` and `SynchronizedRecorderArray` form a useful but complex API. The method `calculate_tdoa_residuals` at module level (not a method of any class) is inconsistently placed.
- **Recommendation:** Move `calculate_tdoa_residuals` to be a method of `SpatialEvent` for better discoverability.

---

## 8. Deprecations for v1.0.0

The following items should be removed in v1.0.0:
- `CNN` alias (or clearly mark as deprecated)
- `overlap_fraction` parameter (replaced by `clip_overlap_fraction`)
- `unpickle` parameter in `CNN.load()` (rename to `allow_pickle`)

---

## 9. Minor Issues Fixed in This PR

The following bugs and inconsistencies were corrected as part of this docstring review:
- Duplicate `frequency_mask` function definition in `preprocess/action_functions.py` (removed)
- Outdated `clip_times` parameter in `BasePreprocessor.forward()` docstring (removed)
- Wrong parameter name `receiver_location` → `receiver_locations` in `spatial_event.py`
- Wrong parameter name `path` → `source` in `AudioSample.__init__` docstring
- `bypass_augmentations` listed in `wandb_table` docstring but not a parameter (removed)
- Incorrect InceptionV3 input size listed as 229×229 (corrected to 299×299)
- Wrong Args section for `change_conv2d_channels` (had `num_classes`/`freeze_feature_extractor` instead of actual params)
- Multiple spelling corrections: `reciever`→`receiver`, `recieved`→`received`, `rollof`→`rolloff`, `saple`→`sample`, etc.
- Non-standard `Inputs:`/`Outputs:` headers in `linear_scale` (corrected to `Args:`/`Returns:`)
- `args:` lowercase header in `data_selection.py` (corrected to `Args:`)
- `Return:` → `Returns:` in `loss.py`
