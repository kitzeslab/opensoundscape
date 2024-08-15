import pytest
import pandas as pd
from pathlib import Path
import shutil

from opensoundscape.ml import lightning
from opensoundscape.utils import make_clip_df


@pytest.fixture()
def train_df():
    return pd.DataFrame(
        index=["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"],
        data=[[0, 1], [1, 0]],
    )


@pytest.fixture()
def train_df_clips(train_df):
    clip_df = make_clip_df(train_df.index.values, clip_duration=1.0)
    clip_df["class0"] = [0] * 10 + [1] * 10
    return clip_df


@pytest.fixture()
def model(train_df_clips):
    model = lightning.LightningSpectrogramModule(
        architecture="resnet18", classes=train_df_clips.columns, sample_duration=1.0
    )
    return model


@pytest.fixture()
def model_save_dir(request):
    path = Path("tests/lightning_ckpts")
    path.parent.mkdir(exist_ok=True)

    # always delete this at the end
    def fin():
        shutil.rmtree(path)

        # lightning logs to a folder in the current dir by default
        # might want to change this to log to save_dir instead?
        lightning_log_dir = Path("lightning_logs")
        if lightning_log_dir.exists():
            shutil.rmtree(lightning_log_dir)

    request.addfinalizer(fin)

    return path


def test_lightning_spectrogram_module_init(model):
    assert model.hparams["architecture"] == "resnet18"
    assert model.hparams["classes"] == ["class0"]
    assert model.hparams["sample_duration"] == 1.0


def test_lightning_spectrogram_module_save_load(model, model_save_dir):
    p = f"{model_save_dir}/temp.ptl"
    model.preprocessor.sample_duration = 5
    model.save(p)
    m2 = lightning.LightningSpectrogramModule.load_from_checkpoint(p)
    assert m2.preprocessor.sample_duration == 5


def test_lightning_spectrogram_module_train(model, train_df_clips, model_save_dir):
    model.fit_with_trainer(
        train_df=train_df_clips,
        validation_df=train_df_clips,
        epochs=1,
        batch_size=8,
        accelerator="auto",
        save_path=model_save_dir,
    )


def test_lightning_spectrogram_module_predict(model, train_df_clips):
    preds = model.predict_with_trainer(
        samples=train_df_clips,
        num_workers=2,
        batch_size=8,
        lightning_trainer_kwargs=dict(accelerator="auto"),
    )
    assert preds.shape == (20, 1)
