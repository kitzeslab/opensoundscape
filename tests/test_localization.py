import numpy as np
import pandas as pd
import pytest
import math
import datetime
import pytz

from opensoundscape import localization
from opensoundscape.localization import localization_algorithms
from opensoundscape.audio import Audio


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
def audiomoth_gps_files():
    return ("tests/audio/audiomoth_gps.wav", "tests/audio/audiomoth_gps.csv")


@pytest.fixture()
def predictions(predictions_csv):
    predictions = pd.read_csv(predictions_csv, index_col=[0, 1, 2])

    # add start timestamps manually since they won't be parsed from these audio files
    predictions["start_timestamp"] = datetime.datetime(
        2021, 9, 24, 6, 52, 0, tzinfo=pytz.UTC
    )
    predictions = predictions.reset_index().set_index(
        ["file", "start_time", "end_time", "start_timestamp"]
    )
    return predictions


@pytest.fixture()
def predictions_no_timezone(predictions_csv):
    """current API will raise an error since datetime is not timezone aware"""
    predictions = pd.read_csv(predictions_csv, index_col=[0, 1, 2])
    predictions["start_timestamp"] = datetime.datetime(2021, 9, 24, 6, 52, 0)
    predictions = predictions.reset_index().set_index(
        ["file", "start_time", "end_time", "start_timestamp"]
    )
    return predictions


@pytest.fixture()
def LOCA_2021_aru_coords():
    return pd.read_csv("tests/csvs/LOCA_2021_aru_coords.csv", index_col=0)


@pytest.fixture()
def LOCA_2021_detections_w_datetimes():
    dets = pd.read_csv("tests/csvs/LOCA_2021_detections.csv", index_col=[0, 1, 2])
    # change microseconds to check this actually gets used
    dets["start_timestamp"] = datetime.datetime(
        2021, 9, 24, 6, 52, 0, 1, tzinfo=pytz.UTC
    )
    dets = dets.reset_index().set_index(
        ["file", "start_time", "end_time", "start_timestamp"]
    )
    return dets


@pytest.fixture()
def LOCA_2021_detections():
    return pd.read_csv("tests/csvs/LOCA_2021_detections.csv", index_col=[0, 1, 2])


@pytest.fixture()
def LOCA_2021_detections_different_file_start_times():
    return pd.read_csv(
        "tests/csvs/LOCA_2021_detections_different_starts.csv", index_col=[0, 1, 2]
    )


def close(x, y, tol):
    return (x < y + tol) and (x > y - tol)


def test_cal_speed_of_sound():
    assert close(localization_algorithms.calc_speed_of_sound(20), 343, 1)


def test_lorentz_ip_3():
    assert localization_algorithms.lorentz_ip([1, 1, 2], [1, 1, 2]) == -2


def test_lorentz_ip_4():
    assert localization_algorithms.lorentz_ip([1, 1, 1, 2], [1, 1, 1, 2]) == -1


def test_lorentz_ip_self():
    assert localization_algorithms.lorentz_ip([1, 1, 1, 2]) == -1


def test_travel_time():
    source = [0, 0, 0]
    receiver = [0, 0, 1]
    assert close(
        localization_algorithms.travel_time(source, receiver, 343), 1 / 343, 0.0001
    )


def test_soundfinder_localize_2d():
    reciever_locations = [[0, 0], [0, 20], [20, 20], [20, 0]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization_algorithms.soundfinder_localize(
        reciever_locations,
        arrival_times,
        speed_of_sound=343,
    )
    assert close(np.linalg.norm(np.array(estimate[0:2]) - np.array([10, 10])), 0, 0.01)


def test_soundfinder_3d():
    reciever_locations = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization_algorithms.soundfinder_localize(
        reciever_locations,
        arrival_times,
        speed_of_sound=343,
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_soundfinder_lstsq():
    # currently not implemented
    reciever_locations = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    with pytest.raises(NotImplementedError):
        estimate = localization_algorithms.soundfinder_localize(
            reciever_locations, arrival_times, invert_alg="lstsq", speed_of_sound=343
        )
    # assert close(
    #     np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    # )


def test_soundfinder_nocenter():
    reciever_locations = [[100, 0, 0], [100, 20, 1], [120, 20, -1], [120, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization_algorithms.soundfinder_localize(
        reciever_locations,
        arrival_times,
        center=False,  # True for original Sound Finder behavior
        speed_of_sound=343,
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([110, 10, 0])), 0, 0.1
    )


def test_gillette_localize_raises():
    reciever_locations = [[100, 0], [100, 20], [120, 20], [120, 0]]
    arrival_times = [1, 1, 1, 1]

    # check this raises a ValueError because none of the arrival times are zero
    with pytest.raises(ValueError):
        localization_algorithms.gillette_localize(
            reciever_locations, arrival_times, speed_of_sound=343
        )


def test_gillette_localize_2d():
    np.random.seed(0)
    receiver_locations = np.array([[0, 0], [0, 20], [20, 20], [20, 0], [10, 10]])
    sound_source = np.random.rand(2) * 20
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_locations - sound_source, axis=1) / speed_of_sound
    )
    tdoas = time_of_flight - np.min(time_of_flight)

    estimated_pos = localization_algorithms.gillette_localize(
        receiver_locations, tdoas, speed_of_sound=speed_of_sound
    )

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

        estimated_pos = localization_algorithms.gillette_localize(
            receiver_locations, tdoas, speed_of_sound=speed_of_sound
        )

        assert np.allclose(estimated_pos, sound_source, atol=2.5)


def test_soundfinder_nopseudo():
    reciever_locations = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    estimate = localization_algorithms.soundfinder_localize(
        reciever_locations,
        arrival_times,
        invert_alg="gps",  # options: 'lstsq', 'gps'
        center=True,  # True for original Sound Finder behavior
        pseudo=False,  # False for original Sound Finder
        speed_of_sound=343,
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

        estimated_pos = localization_algorithms.least_squares_localize(
            receiver_locations, tdoas, speed_of_sound=speed_of_sound
        )

        assert np.allclose(estimated_pos, sound_source, atol=2.5)


def test_asserts_localized_timestamps(file_coords_csv, predictions_no_timezone):
    file_coords = pd.read_csv(file_coords_csv, index_col=0)

    array = localization.SynchronizedRecorderArray(file_coords=file_coords)
    with pytest.raises(ValueError):
        array.localize_detections(
            detections=predictions_no_timezone,
            min_n_receivers=4,
            max_receiver_dist=100,
            localization_algorithm="least_squares",
            return_unlocalized=True,
        )


def test_localization_pipeline(file_coords_csv, predictions):
    file_coords = pd.read_csv(file_coords_csv, index_col=0)

    array = localization.SynchronizedRecorderArray(file_coords=file_coords)
    position_estimates, _ = array.localize_detections(
        detections=predictions,
        min_n_receivers=4,
        max_receiver_dist=100,
        localization_algorithm="least_squares",
        return_unlocalized=True,
    )
    # the audio files were generated according to the "true" event location:
    true_x = 10
    true_y = 15
    assert len(position_estimates) == 5

    for position in position_estimates:
        assert math.isclose(position.location_estimate[0], true_x, abs_tol=2)
        assert math.isclose(position.location_estimate[1], true_y, abs_tol=2)

    # test load_audio_segments, loading with 1s before and after the event start/end

    with pytest.warns(UserWarning):  # warning for extending beyond edges of audio
        audio_list = position_estimates[0].load_aligned_audio_segments(
            start_offset=1, end_offset=1
        )
    assert len(audio_list) == 5
    # event is 1s long, so we should have 3s total (slightly less for others
    # due to tdoa offsets and extending beyond file edges)
    assert np.isclose(audio_list[0].duration, 3, atol=1e-5)


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

    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    localized_events, _ = array.localize_detections(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        localization_algorithm="gillette",
        bandpass_ranges={"zeep": (7000, 10000)},
        return_unlocalized=True,
    )

    bad_file = "tests/audio/.wav"
    # check that the bad file has been dropped from the event
    for event in localized_events:
        assert bad_file not in event.receiver_files
        assert len(event.receiver_files) == 6


def test_SpatialEvent_estimate_delays(LOCA_2021_aru_coords):
    # Test ensure that SpatialEvent_estimate_delays returns what is expected
    max_delay = 0.04
    receiver_start_time_offsets = [0.2] * (len(LOCA_2021_aru_coords) - 1) + [0.1]
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
        [0, 0.0325, -0.002, 0.0316, -0.0086, 0.024, 0.024]
    )  # with reference receiver LOCA_2021_3...
    assert np.allclose(event.tdoas, true_TDOAS, atol=0.01)


def test_SpatialEvent_estimate_delays_auto_timestamps(LOCA_2021_aru_coords):
    # test localization of SpatialEvent when it attempts to find
    # individual file start timestamps from the audio files themselves and start_timestamp
    # instead of user providing receiver_start_time_offsets
    max_delay = 0.04
    duration = 0.3
    cc_filter = "phat"
    bandpass_range = (5000, 10000)

    event = localization.SpatialEvent(
        receiver_files=LOCA_2021_aru_coords.index,
        receiver_locations=LOCA_2021_aru_coords.values,
        max_delay=max_delay,
        duration=duration,
        class_name="zeep",
        bandpass_range=bandpass_range,
        cc_filter=cc_filter,
        start_timestamp=datetime.datetime(
            2021, 9, 24, 6, 52, 0, 200_000, tzinfo=pytz.UTC
        ),
    )

    # check that the delays are what we expect
    event._estimate_delays()
    true_TDOAS = np.array(
        [0, 0.0325, -0.002, 0.0316, -0.0086, 0.024, 0.024]
    )  # with reference receiver LOCA_2021_3...
    assert np.allclose(event.tdoas, true_TDOAS, atol=0.01)


def test_localization_pipeline_parallelized(LOCA_2021_aru_coords, LOCA_2021_detections):
    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    localized_events = array.localize_detections(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        localization_algorithm="gillette",
        cc_filter="phat",
        bandpass_ranges={"zeep": (7000, 10000)},
        num_workers=2,
    )

    true_TDOAS = np.array(
        [0, 0.0325, -0.002, 0.0316, -0.0086, 0.024]
    )  # with reference receiver LOCA_2021_3...

    assert len(localized_events) == 6
    checked = False
    for event in localized_events:
        if event.receiver_files[0] == "tests/audio/LOCA_2021_09_24_652_3.wav":
            assert np.allclose(event.tdoas, true_TDOAS, atol=0.01)
            checked = True
    assert checked


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
            num_workers=1,
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


def test_create_candidate_events_finds_timestamps(
    LOCA_2021_detections, LOCA_2021_aru_coords
):
    # when creating candidate events, start_timestamp is obtained from metadata if not included in detections df
    # will fail if recording_start_time not in metadata parsed from file
    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    candidate_events = array.create_candidate_events(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        cc_threshold=0,
        bandpass_ranges={"zeep": (7000, 10000)},
        cc_filter="phat",
    )
    for i, event in enumerate(candidate_events):
        assert event.start_timestamp.to_pydatetime() == datetime.datetime(
            2021, 9, 24, 6, 52, 0, tzinfo=pytz.UTC
        ) + datetime.timedelta(
            seconds=LOCA_2021_detections.reset_index().iloc[i]["start_time"]
        )


def test_create_candidate_events_provided_timestamps(
    LOCA_2021_detections_w_datetimes, LOCA_2021_aru_coords
):
    # the LOCA_2021_detections_w_datetimes dataframe has a fourth multi-index level "start_timestamp"
    # which is used to set the start_timestamp of the candidate events, rather than trying to parse from the audio files

    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    candidate_events = array.create_candidate_events(
        detections=LOCA_2021_detections_w_datetimes,
        min_n_receivers=4,
        max_receiver_dist=30,
        cc_threshold=0,
        bandpass_ranges={"zeep": (7000, 10000)},
        cc_filter="phat",
    )
    for i, event in enumerate(candidate_events):
        assert (
            event.start_timestamp
            == LOCA_2021_detections_w_datetimes.reset_index().iloc[i]["start_timestamp"]
        )

    # test that the events can be localized
    localized_events = array.localize_detections(
        detections=LOCA_2021_detections_w_datetimes,
        localization_algorithm="gillette",
        cc_filter="phat",
        num_workers=1,
        max_receiver_dist=30,
        min_n_receivers=4,
    )
    assert len(localized_events) == 6


def test_localize_from_files_with_different_start_times(
    LOCA_2021_aru_coords, LOCA_2021_detections_different_file_start_times
):
    # test that the localize_detections method can handle detections from different files with different start times
    # and that the start times are correctly used to set the start_timestamp of the candidate events
    array = localization.SynchronizedRecorderArray(
        file_coords=LOCA_2021_aru_coords,
    )
    localized_events = array.localize_detections(
        detections=LOCA_2021_detections_different_file_start_times,
        localization_algorithm="gillette",
        cc_filter="phat",
        num_workers=1,
        max_receiver_dist=30,
        min_n_receivers=4,
        # cc_threshold=0,
    )
    assert len(localized_events) == 6
    e = localized_events[0]
    # last file starts 0.1 sec later, so offset from beginning of file to event is 0.1 sec less than others
    assert (e.receiver_start_time_offsets == [0.2, 0.2, 0.2, 0.2, 0.2, 0.1]).all()
    assert e.start_timestamp.to_pydatetime() == datetime.datetime(
        2021, 9, 24, 6, 52, 0, 200_000, tzinfo=pytz.UTC
    )
    # TODO: check that the position estimate is correct


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


def test_cc_threshold_mask_receivers(LOCA_2021_aru_coords, LOCA_2021_detections):
    """when some receivers are filtered out based on cc threshold,
    they should not be included in attributes like .receiver_start_time_offsets
    """
    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    localized_positions, unlocalized_events = array.localize_detections(
        detections=LOCA_2021_detections,
        min_n_receivers=4,
        max_receiver_dist=30,
        localization_algorithm="gillette",
        cc_filter="phat",
        cc_threshold=0.05,
        return_unlocalized=True,
    )
    e = localized_positions[0]
    assert len(e.receiver_start_time_offsets) == 4
    assert e.receiver_start_time_offsets[0] == 0.2
    assert len(e.receiver_files) == 4
    assert len(e.receiver_locations) == 4
    assert len(e.tdoas) == 4
    assert len(e.cc_maxs) == 4
    assert len(e.distance_residuals) == 4


def test_spatial_event_to_from_dict(LOCA_2021_aru_coords):
    # test that a SpatialEvent can be serialized to a dictionary and then re-instantiated
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
        start_timestamp=datetime.datetime(
            2021, 9, 24, 6, 52, 0, 200_000, tzinfo=pytz.UTC
        ),
    )

    event_dict = event.to_dict()
    assert isinstance(event_dict, dict)

    new_event = localization.SpatialEvent.from_dict(event_dict)

    assert event.start_timestamp == new_event.start_timestamp
    assert (
        event.receiver_start_time_offsets == new_event.receiver_start_time_offsets
    ).all()
    assert (event.receiver_files == new_event.receiver_files).all()
    # compare equality of two arrays that can contain nan
    assert np.array_equal(
        event.receiver_locations, new_event.receiver_locations, equal_nan=True
    )
    assert event.max_delay == new_event.max_delay
    assert event.duration == new_event.duration
    assert event.class_name == new_event.class_name
    assert event.bandpass_range == new_event.bandpass_range
    assert event.cc_filter == new_event.cc_filter


def test_spatial_event_localize_not_enough_receivers():
    # should get empty values but should not raise exception
    # when trying to localize with too few receivers (e.g. 2)
    event = localization.SpatialEvent(
        receiver_files=["file1", "file2"],
        receiver_locations=np.array([[0, 0], [0, 20]]),
        max_delay=0.04,
        receiver_start_time_offsets=np.array([0.2, 0.2]),
        duration=0.3,
        class_name="zeep",
        bandpass_range=(5000, 10000),
        cc_filter="phat",
        start_timestamp=datetime.datetime(
            2021, 9, 24, 6, 52, 0, 200_000, tzinfo=pytz.UTC
        ),
    )
    event.tdoas = np.array([0, 0.0325])
    event.cc_maxs = np.array([1, 0.8])
    position_estimate = event._localize_after_cross_correlation(
        localization_algorithm="gillette"
    )
    assert position_estimate.location_estimate is None
    assert position_estimate.distance_residuals is None


def test_position_estimate_to_from_dict():
    # test that a PositionEstimate can be serialized to a dictionary and then re-instantiated
    position_estimate = localization.PositionEstimate(
        location_estimate=np.array([10, 15]),
        start_timestamp=datetime.datetime(
            2021, 9, 24, 6, 52, 0, 200_000, tzinfo=pytz.UTC
        ),
        class_name="zeep",
        receiver_files=np.array(["file1", "file2"]),
        tdoas=np.array([0, 0.0325]),
        cc_maxs=np.array([1, 0.8]),
        receiver_locations=np.array([[0, 0], [0, 20]]),
        receiver_start_time_offsets=np.array([0.2, 0.2]),
        duration=0.3,
        distance_residuals=np.array([0.1, 0.2]),
    )

    position_estimate_dict = position_estimate.to_dict()
    assert isinstance(position_estimate_dict, dict)

    new_position_estimate = localization.PositionEstimate.from_dict(
        position_estimate_dict
    )

    for attr in [
        "location_estimate",
        "start_timestamp",
        "class_name",
        "receiver_files",
        "tdoas",
        "cc_maxs",
        "receiver_locations",
        "receiver_start_time_offsets",
        "duration",
        "distance_residuals",
    ]:
        val = getattr(position_estimate, attr)
        if isinstance(val, np.ndarray):
            assert np.array_equal(val, getattr(new_position_estimate, attr))
        else:
            assert val == getattr(new_position_estimate, attr)


def test_df_to_positions(LOCA_2021_aru_coords, LOCA_2021_detections):
    # test that a dataframe of detections can be converted to a list of PositionEstimates
    array = localization.SynchronizedRecorderArray(file_coords=LOCA_2021_aru_coords)
    positions = array.localize_detections(
        detections=LOCA_2021_detections,
        localization_algorithm="gillette",
        cc_filter="phat",
        num_workers=1,
        max_receiver_dist=30,
        min_n_receivers=4,
    )
    df = localization.position_estimate.positions_to_df(positions)
    assert isinstance(df, pd.DataFrame)

    # try to recover the events
    recovered_positions = localization.position_estimate.df_to_positions(df)

    for i, event in enumerate(positions):
        assert event.start_timestamp == recovered_positions[i].start_timestamp
        assert (
            event.receiver_start_time_offsets
            == recovered_positions[i].receiver_start_time_offsets
        ).all()
        assert (event.receiver_files == recovered_positions[i].receiver_files).all()
        # compare equality of two arrays that can contain nan
        assert np.array_equal(
            event.receiver_locations,
            recovered_positions[i].receiver_locations,
            equal_nan=True,
        )
        assert event.duration == recovered_positions[i].duration
        assert event.class_name == recovered_positions[i].class_name
        assert (event.tdoas == recovered_positions[i].tdoas).all()
        assert (
            event.location_estimate == recovered_positions[i].location_estimate
        ).all()
        assert (event.cc_maxs == recovered_positions[i].cc_maxs).all()


def test_resample_audiomoth_file_with_pps(audiomoth_gps_files):
    audio_file, pps_file = audiomoth_gps_files
    # create correspondence between GPS timestamps and WAV file sample positions
    pps = pd.read_csv(pps_file, index_col=0)
    samples_timestamps = localization.audiomoth_sync.associate_pps_samples_timestamps(
        pps
    )

    # Resample the audio second-by-second using the GPS timestamps to achieve nominal samping rate
    resampled_audio = localization.audiomoth_sync.correct_sample_rate(
        Audio.from_file(audio_file), samples_timestamps, desired_sr=48000
    )
    assert len(resampled_audio.samples) == 48000 * 2
