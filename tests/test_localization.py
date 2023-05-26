import numpy as np
import pandas as pd
import pytest
import math

from opensoundscape import localization


@pytest.fixture()
def aru_files():
    return [f"tests/audio/aru_{i}.wav" for i in range(1, 6)]


@pytest.fixture()
def file_coords_csv():
    return "tests/csvs/aru_coords.csv"


@pytest.fixture()
def predictions_csv():
    return "tests/csvs/localizer_preds.csv"


def close(x, y, tol):
    return (x < y + tol) and (x > y - tol)


def test_cal_speed_of_sound():
    assert close(localization.calc_speed_of_sound(20), 343, 1)


def test_lorentz_ip_3():
    assert localization.lorentz_ip([1, 1, 2], [1, 1, 2]) == -2


def test_lorentz_ip_4():
    assert localization.lorentz_ip([1, 1, 1, 2], [1, 1, 1, 2]) == -1


def test_lorentz_ip_self():
    assert localization.lorentz_ip([1, 1, 1, 2]) == -1


def test_travel_time():
    source = [0, 0, 0]
    receiver = [0, 0, 1]
    assert close(localization.travel_time(source, receiver, 343), 1 / 343, 0.0001)


def test_soundfinder_localize_2d():
    reciever_positions = [[0, 0], [0, 20], [20, 20], [20, 0]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_positions,
        arrival_times,
    )
    assert close(np.linalg.norm(np.array(estimate[0:2]) - np.array([10, 10])), 0, 0.01)


def test_soundfinder_3d():
    reciever_positions = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_positions,
        arrival_times,
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_soundfinder_lstsq():
    # currently not implemented
    reciever_positions = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    with pytest.raises(NotImplementedError):
        estimate = localization.soundfinder_localize(
            reciever_positions, arrival_times, invert_alg="lstsq"
        )
    # assert close(
    #     np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    # )


def test_soundfinder_nocenter():
    reciever_positions = [[100, 0, 0], [100, 20, 1], [120, 20, -1], [120, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_positions,
        arrival_times,
        center=False,  # True for original Sound Finder behavior
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([110, 10, 0])), 0, 0.1
    )


def test_gillette_localize_raises():
    reciever_positions = [[100, 0], [100, 20], [120, 20], [120, 0]]
    arrival_times = [1, 1, 1, 1]

    # check this raises a ValueError because none of the arrival times are zero
    with pytest.raises(ValueError):
        localization.gillette_localize(reciever_positions, arrival_times)


def test_gillette_localize_2d():
    np.random.seed(0)
    receiver_positions = np.array([[0, 0], [0, 20], [20, 20], [20, 0], [10, 10]])
    sound_source = np.random.rand(2) * 20
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_positions - sound_source, axis=1) / speed_of_sound
    )
    tdoas = time_of_flight - np.min(time_of_flight)

    estimated_pos = localization.gillette_localize(receiver_positions, tdoas)

    assert np.allclose(estimated_pos, sound_source, rtol=0.1)


def test_gillette_localize_3d():
    receiver_positions = np.array(
        [[0, 0, 10], [0, 20, 1], [20, 20, -1], [20, 0, 0.1], [10, 10, 0], [5, 5, 5]]
    )
    sound_source = np.array([10, 12, 2])
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_positions - sound_source, axis=1) / speed_of_sound
    )

    # localize with each receiver as reference:
    for ref_index in range(len(time_of_flight)):
        tdoas = time_of_flight - time_of_flight[ref_index]

        estimated_pos = localization.gillette_localize(receiver_positions, tdoas)

        assert np.allclose(estimated_pos, sound_source, atol=2.5)


def test_soundfinder_nopseudo():
    reciever_positions = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_positions,
        arrival_times,
        invert_alg="gps",  # options: 'lstsq', 'gps'
        center=True,  # True for original Sound Finder behavior
        pseudo=False,  # False for original Sound Finder
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_localizer(file_coords_csv, predictions_csv):
    file_coords = pd.read_csv(file_coords_csv, index_col=0)
    preds = pd.read_csv(predictions_csv, index_col=[0, 1, 2])
    array = localization.SynchronizedRecorderArray(file_coords=file_coords)
    localized_events, _ = array.localize_detections(
        detections=preds,
        min_n_receivers=4,
        max_receiver_dist=100,
        localization_algorithm="gillette",
    )
    # the audio files were generated according to the "true" event position:
    true_x = 10
    true_y = 15
    assert len(localized_events) == 5

    for event in localized_events:
        assert math.isclose(event.position_estimate[0], true_x, abs_tol=2)
        assert math.isclose(event.position_estimate[1], true_y, abs_tol=2)
