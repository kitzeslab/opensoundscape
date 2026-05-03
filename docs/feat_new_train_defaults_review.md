# feat_new_train_defaults Review

This note summarizes the direct branch-authored changes on `feat_new_train_defaults`, the small compatibility fixes applied during review, and the remaining failing tests that still need follow-up.

## What Changed On The Branch

- Training/inference defaults were refactored around `SpectrogramClassifier`, `SafeAudioDataloader`, and the preprocessing stack.
- The preprocessing pipeline gained new augmentation/default behavior, including overlay handling and a default bandpass path in `SpectrogramPreprocessor`.
- Dataset ingestion was expanded to support more dataframe shapes and clip generation helpers.
- Training features were added around early stopping, best-model reloading, profiling, and optimizer/scheduler management.
- Several docs and tutorials were updated to match the new training flow.

## Fixes Applied During Review

- Preserved clip dataframe indices in `_ingest_samples_argument()` instead of resetting them and then trying to re-index missing columns.
- Hardened `_check_first_path()` so it handles both scalar file indexes and multi-index clip rows.
- Removed the `sample_duration` constructor-only field from the runtime overlay function call.
- Restored the default spectrogram bandpass insertion when a sample rate is known.
- Restored backward-compatible `sample_rate=None` defaults on `SpectrogramClassifier`/`SpectrogramModule` so older call sites still construct models.

## Current Test Status

Focused suites improved from a broad failure cluster to `81 passed, 26 failed`.

## Remaining Failing Tests

### Likely code issues

- `tests/test_cnn.py::test_classifier_custom_lr`
  - The optimizer is not preserving the requested base LR once classifier-specific LR logic runs.

- `tests/test_cnn.py::test_reset_or_keep_optimizer_and_scheduler`
  - Optimizer state is not being preserved in the way the test expects after a no-op retrain.

- `tests/test_cnn.py::test_predict_on_empty_list`
  - Empty predictions still return the clip-style index layout instead of the historical empty-frame layout expected by the test.

- `tests/test_cnn.py::test_predict_all_arch_4ch`
- `tests/test_cnn.py::test_predict_all_arch_1ch`
  - Architecture/device compatibility appears incomplete for some backends, especially on MPS with certain torchvision models.

- `tests/test_cnn.py::test_prediction_overlap`
- `tests/test_cnn.py::test_predict_splitting_short_file`
- `tests/test_cnn.py::test_train_predict_architecture`
- `tests/test_cnn.py::test_embed`
- `tests/test_cnn.py::test_embed_no_avgpool`
- `tests/test_cnn.py::test_call_with_targets`
  - These still look like API/shape adaptation gaps around the refactored prediction/embed/call paths.

- `tests/test_cnn.py::test_generate_cam_all_architectures`
- `tests/test_cnn.py::test_save_onnx`
  - Export/path coverage is probably still behind the new preprocessing/model defaults.

- `tests/test_datasets.py::test_overlay_*`
- `tests/test_datasets.py::test_spec_preprocessor_overlay`
  - Overlay augmentation still needs validation against the new ingestion and label-update flow.

### Unclear / needs more investigation

- `tests/test_datasets.py::test_audio_file_dataset_no_reshape`
  - The current preprocessor pipeline still produces a spectrogram shape that disagrees with the legacy expectation; this may be a genuine semantic change or an outdated assertion.

- `tests/test_datasets.py::test_audio_splitting_dataset_overlap`
  - The failure points to clip splitting behavior, but the exact regression needs another pass against the split-time helper and the dataset wrapper.

- `tests/test_cnn.py::test_predict_posixpath_missing_files`
  - The failure is path-handling related, but the right fix depends on whether the new ingestion rules intentionally tightened missing-file validation.

## Recommended Next Steps

1. Fix optimizer/scheduler retention first, because those failures are isolated and likely to unblock a few training tests.
2. Then inspect the embed/predict/cam code paths for stale assumptions about clip layout and empty results.
3. Finally, decide whether `test_audio_file_dataset_no_reshape` is asserting legacy behavior that should be restored or a shape expectation that should be updated.