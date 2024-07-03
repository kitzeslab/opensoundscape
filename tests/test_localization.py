import numpy as np
import pandas as pd
import pytest
import math
import datetime

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


@pytest.fixture()
def predictions(predictions_csv):
    predictions = pd.read_csv(predictions_csv, index_col=[0, 1, 2])

    # add start timestamps manually since they won't be parsed from these audio files
    predictions["start_timestamp"] = datetime.datetime(2021, 9, 24, 6, 52, 0)
    predictions = predictions.reset_index().set_index(
        ["file", "start_time", "end_time", "start_timestamp"]
    )
    return predictions


@pytest.fixture()
def LOCA_2021_aru_coords():
    return pd.read_csv("tests/csvs/LOCA_2021_aru_coords.csv", index_col=0)


@pytest.fixture()
def LOCA_2021_detections():
    dets = pd.read_csv("tests/csvs/LOCA_2021_detections.csv", index_col=[0, 1, 2])
    dets["start_timestamp"] = datetime.datetime(2021, 9, 24, 6, 52, 0)
    dets = dets.reset_index().set_index(
        ["file", "start_time", "end_time", "start_timestamp"]
    )
    return dets


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
    reciever_locations = [[0, 0], [0, 20], [20, 20], [20, 0]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_locations,
        arrival_times,
    )
    assert close(np.linalg.norm(np.array(estimate[0:2]) - np.array([10, 10])), 0, 0.01)


def test_soundfinder_3d():
    reciever_locations = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_locations,
        arrival_times,
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_soundfinder_lstsq():
    # currently not implemented
    reciever_locations = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    with pytest.raises(NotImplementedError):
        estimate = localization.soundfinder_localize(
            reciever_locations, arrival_times, invert_alg="lstsq"
        )
    # assert close(
    #     np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    # )


def test_soundfinder_nocenter():
    reciever_locations = [[100, 0, 0], [100, 20, 1], [120, 20, -1], [120, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_locations,
        arrival_times,
        center=False,  # True for original Sound Finder behavior
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([110, 10, 0])), 0, 0.1
    )


def test_gillette_localize_raises():
    reciever_locations = [[100, 0], [100, 20], [120, 20], [120, 0]]
    arrival_times = [1, 1, 1, 1]

    # check this raises a ValueError because none of the arrival times are zero
    with pytest.raises(ValueError):
        localization.gillette_localize(reciever_locations, arrival_times)


def test_gillette_localize_2d():
    np.random.seed(0)
    receiver_locations = np.array([[0, 0], [0, 20], [20, 20], [20, 0], [10, 10]])
    sound_source = np.random.rand(2) * 20
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_locations - sound_source, axis=1) / speed_of_sound
    )
    tdoas = time_of_flight - np.min(time_of_flight)

    estimated_pos = localization.gillette_localize(receiver_locations, tdoas)

    assert np.allclose(estimated_pos, sound_source, rtol=0.1)


def test_gillette_localize_3d():
    receiver_locations = np.array(
        [[0, 0, 10], [0, 20, 1], [20, 20, -1], [20, 0, 0.1], [10, 10, 0], [5, 5, 5]]
    )
    sound_source = np.array([10, 12, 2])
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_locations - sound_source, axis=1) / speed_of_sound
    )

    # localize with each receiver as reference:
    for ref_index in range(len(time_of_flight)):
        tdoas = time_of_flight - time_of_flight[ref_index]

        estimated_pos = localization.gillette_localize(receiver_locations, tdoas)

        assert np.allclose(estimated_pos, sound_source, atol=2.5)


def test_soundfinder_nopseudo():
    reciever_locations = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization.soundfinder_localize(
        reciever_locations,
        arrival_times,
        invert_alg="gps",  # options: 'lstsq', 'gps'
        center=True,  # True for original Sound Finder behavior
        pseudo=False,  # False for original Sound Finder
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_least_squares_optimize():
    receiver_locations = np.array(
        [[0, 0, 10], [0, 20, 1], [20, 20, -1], [20, 0, 0.1], [10, 10, 0], [5, 5, 5]]
    )
    sound_source = np.array([10, 12, 2])
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_locations - sound_source, axis=1) / speed_of_sound
    )

    # localize with each receiver as reference:
    for ref_index in range(len(time_of_flight)):
        tdoas = time_of_flight - time_of_flight[ref_index]

        estimated_pos = localization.least_squares_localize(receiver_locations, tdoas)

        assert np.allclose(estimated_pos, sound_source, atol=2.5)


def test_localization_pipeline(file_coords_csv, predictions):
    file_coords = pd.read_csv(file_coords_csv, index_col=0)

    array = localization.SynchronizedRecorderArray(file_coords=file_coords)
    localized_events, _ = array.localize_detections(
        detections=predictions,
        min_n_receivers=4,
        max_receiver_dist=100,
        localization_algorithm="least_squares",
        return_unlocalized=True,
    )
    # the audio files were generated according to the "true" event location:
    true_x = 10
    true_y = 15
    assert len(localized_events) == 5

    for event in localized_events:
        assert math.isclose(event.location_estimate[0], true_x, abs_tol=2)
        assert math.isclose(event.location_estimate[1], true_y, abs_tol=2)


def test_localization_pipeline_real_audio(LOCA_2021_aru_coords, LOCA_2021_detections):

    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    localized_events = array.localize_detections(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        localization_algorithm="gillette",
        cc_filter="phat",
        bandpass_ranges={"zeep": (7000, 10000)},
    )

    true_TDOAS = np.array(
        [0, 0.0325, -0.002, 0.0316, -0.0086, 0.024]
    )  # with reference receiver LOCA_2021_3...

    for event in localized_events:
        if event.receiver_files[0] == "tests/audio/LOCA_2021_09_24_652_3.wav":
            assert np.allclose(event.tdoas, true_TDOAS, atol=0.01)


def test_unlocalized_events(file_coords_csv, predictions):
    file_coords = pd.read_csv(file_coords_csv, index_col=0)
    array = localization.SynchronizedRecorderArray(file_coords=file_coords)
    localized_events, unlocalized_events = array.localize_detections(
        detections=predictions,
        min_n_receivers=4,
        cc_threshold=100,  # too high. Spatial events will all be unlocalized.
        max_receiver_dist=100,
        localization_algorithm="gillette",
        return_unlocalized=True,
    )
    assert len(localized_events) == 0
    assert len(unlocalized_events) > 1


def test_SynchronizedRecorderArray_SpatialEvents_not_generated(
    file_coords_csv, predictions
):
    # Tests that the SynchronizedRecorderArray will not return any SpatialEvents if
    # min_n_receivers is set too high.
    file_coords = pd.read_csv(file_coords_csv, index_col=0)
    array = localization.SynchronizedRecorderArray(file_coords=file_coords)
    localized_events, unlocalized_events = array.localize_detections(
        detections=predictions,
        min_n_receivers=10,  # too high. No SpatialEvents will be outputted.
        cc_threshold=0,
        max_receiver_dist=100,
        localization_algorithm="gillette",
        return_unlocalized=True,
    )
    assert len(localized_events) == 0
    assert len(unlocalized_events) == 0


def test_localization_pipeline_real_audio_edge_case(
    LOCA_2021_aru_coords, LOCA_2021_detections
):
    # this test ensures that the localization pipeline does not fail when one of the files
    # in the detections dataframe is actually too shorter
    # i.e. the file is shorter than the minimum length needed for cross correlation

    # TODO: add tests for automatically finding timestamps - need audio with recording start time metadata

    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    localized_events, _ = array.localize_detections(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        localization_algorithm="gillette",
        bandpass_ranges={"zeep": (7000, 10000)},
        return_unlocalized=True,
    )

    bad_file = "tests/audio/veryshort.wav"
    # check that the bad file has been dropped from the event
    for event in localized_events:
        assert bad_file not in event.receiver_files
        assert len(event.receiver_files) == 6


def test_SpatialEvent_estimate_delays(LOCA_2021_aru_coords, LOCA_2021_detections):
    # Test ensure that SpatialEvent_estimate_delays returns what is expected

    LOCA_2021_aru_coords = pd.read_csv(LOCA_2021_aru_coords, index_col=0)

    max_delay = 0.04
    receiver_start_time_offsets = [0.2] * len(LOCA_2021_aru_coords)
    duration = 0.3
    cc_filter = "phat"
    bandpass_range = (5000, 10000)

    event = localization.SpatialEvent(
        receiver_files=LOCA_2021_aru_coords.index,
        receiver_locations=LOCA_2021_aru_coords.values,
        max_delay=max_delay,
        receiver_start_time_offsets=receiver_start_time_offsets,
        duration=duration,
        class_name="zeep",
        bandpass_range=bandpass_range,
        cc_filter=cc_filter,
    )

    # check that the delays are what we expect
    event._estimate_delays()
    true_TDOAS = np.array(
        [0, 0.0325, -0.002, 0.0316, -0.0086, 0.024]
    )  # with reference receiver LOCA_2021_3...
    assert np.allclose(event.tdoas, true_TDOAS, atol=0.01)


# TODO: add similar test where start_timestamp is provided instead of receiver_start_time_offsets


def test_localization_pipeline_parallelized(LOCA_2021_aru_coords, LOCA_2021_detections):
    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    localized_events = array.localize_detections(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        localization_algorithm="gillette",
        cc_filter="phat",
        bandpass_ranges={"zeep": (7000, 10000)},
        num_workers=4,
    )

    true_TDOAS = np.array(
        [0, 0.0325, -0.002, 0.0316, -0.0086, 0.024]
    )  # with reference receiver LOCA_2021_3...

    for event in localized_events:
        if event.receiver_files[0] == "tests/audio/LOCA_2021_09_24_652_3.wav":
            assert np.allclose(event.tdoas, true_TDOAS, atol=0.01)


def test_localization_pipeline_cc_filters(LOCA_2021_aru_coords, LOCA_2021_detections):
    ## Test that the different filters work, and are returning DIFFERENT cc values
    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)

    cc_scores = {}
    for cc_filter in ["phat", "cc_norm", "roth"]:
        localized_events = array.localize_detections(
            detections=LOCA_2021_detections,
            min_n_receivers=4,
            max_receiver_dist=30,
            localization_algorithm="gillette",
            cc_filter=cc_filter,
            bandpass_ranges={"zeep": (7000, 10000)},
            num_workers=4,
        )
        for event in localized_events:
            if event.receiver_files[0] == "tests/audio/LOCA_2021_09_24_652_3.wav":
                cc_scores[cc_filter] = event.cc_maxs  # save the cc scores
        # check that the cc scores are different
    assert (
        not np.allclose(cc_scores["phat"], cc_scores["cc_norm"], atol=0.001)
        and not np.allclose(cc_scores["phat"], cc_scores["roth"], atol=0.001)
        and not np.allclose(cc_scores["cc_norm"], cc_scores["roth"], atol=0.001)
    )


# TODO: need new tests for automatically extracting time stamps both in SynchronizedRecorderArray.create_candidate_events and
# #SpatialEvent.localize(start_timetsamp...)

# def test_spatial_event_timestamps(LOCA_2021_aru_coords, LOCA_2021_detections):
#     """Check that the timestamps of the spatial events are correct"""
#     import datetime

#     array = localization.SynchronizedRecorderArray(
#         file_coords=LOCA_2021_aru_coords,
#     )

#     localized_events = array.localize_detections(
#         detections=LOCA_2021_detections,
#         min_n_receivers=4,
#         max_receiver_dist=30,
#         localization_algorithm="gillette",
#         cc_filter="phat",
#         bandpass_ranges={"zeep": (7000, 10000)},
#         num_workers=4,
#     )
#     for event in localized_events:
#         assert event.start_timestamp == datetime.datetime(
#             2021, 9, 24, 6, 52, 0
#         ) + datetime.timedelta(seconds=event.start_time)


def test_localize_too_few_receivers(LOCA_2021_aru_coords, LOCA_2021_detections):
    """Check that the localization pipeline does not return a position estimate
    when there are too few receivers left after filtering by cc_threshold

    events that originally had enough recorders, but after filtering by cc_threshold
    have too few recorders, should be returned as unlocalized events
    """
    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    localized_events, unlocalized_events = array.localize_detections(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        localization_algorithm="gillette",
        cc_filter="phat",
        cc_threshold=100,
        return_unlocalized=True,
    )
    assert len(localized_events) == 0
    assert len(unlocalized_events) == 6
