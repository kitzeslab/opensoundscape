import numpy as np
from opensoundscape import localization


@pytest.fixture()
def localizer_files():
    return [f"tests/localizer_tests/aru_{i}.wav" for i in range(1, 6)]


# @pytest.fixture()
# def aru_1():
#     return "tests/localizer_tests/aru_1.wav"

# @pytest.fixture()
# def aru_2():
#     return "tests/localizer_tests/aru_2.wav"

# @pytest.fixture()
# def aru_3():
#     return "tests/localizer_tests/aru_3.wav"

# @pytest.fixture()
# def aru_4():
#     return "tests/localizer_tests/aru_4.wav"

# @pytest.fixture()
# def aru_5():
#     return "tests/localizer_tests/aru_5.wav"


@pytest.fixture()
def aru_coords():
    return "tests/localizer_tests/aru_coords.csv"


@pytest.fixture()
def predictions():
    return "tests/localizer_tests/preds.csv"


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


def test_soundfinder_2d():
    reciever_positions = [[0, 0], [0, 20], [20, 20], [20, 0]]
    arrival_times = [1, 1, 1, 1]
    invert_alg = ("gps",)  # options: 'lstsq', 'gps'
    center = (True,)  # True for original Sound Finder behavior
    pseudo = (True,)  # False for original Sound Finder
    estimate = localization.soundfinder(
        reciever_positions,
        arrival_times,
        temperature=20.0,  # celcius
        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
        center=center,  # True for original Sound Finder behavior
        pseudo=pseudo,  # False for original Sound Finder
    )
    assert close(np.linalg.norm(np.array(estimate[0:2]) - np.array([10, 10])), 0, 0.01)


def test_soundfinder_3d():
    reciever_positions = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    invert_alg = ("gps",)  # options: 'lstsq', 'gps'
    center = (True,)  # True for original Sound Finder behavior
    pseudo = (True,)  # False for original Sound Finder
    estimate = localization.soundfinder(
        reciever_positions,
        arrival_times,
        temperature=20.0,  # celcius
        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
        center=center,  # True for original Sound Finder behavior
        pseudo=pseudo,  # False for original Sound Finder
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_soundfinder_lstsq():
    reciever_positions = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    invert_alg = ("lstsq",)  # options: 'lstsq', 'gps'
    center = (True,)  # True for original Sound Finder behavior
    pseudo = (True,)  # False for original Sound Finder
    estimate = localization.soundfinder(
        reciever_positions,
        arrival_times,
        temperature=20.0,  # celcius
        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
        center=center,  # True for original Sound Finder behavior
        pseudo=pseudo,  # False for original Sound Finder
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_soundfinder_nocenter():
    reciever_positions = [[100, 0, 0], [100, 20, 1], [120, 20, -1], [120, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    invert_alg = ("lstsq",)  # options: 'lstsq', 'gps'
    center = (False,)  # True for original Sound Finder behavior
    pseudo = (True,)  # False for original Sound Finder
    estimate = localization.soundfinder(
        reciever_positions,
        arrival_times,
        temperature=20.0,  # celcius
        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
        center=center,  # True for original Sound Finder behavior
        pseudo=pseudo,  # False for original Sound Finder
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([110, 10, 0])), 0, 0.1
    )


def test_soundfinder_nopseudo():
    reciever_positions = [[0, 0, 0], [0, 20, 1], [20, 20, -1], [20, 0, 0.1]]
    arrival_times = [1, 1, 1, 1]
    invert_alg = ("lstsq",)  # options: 'lstsq', 'gps'
    center = (True,)  # True for original Sound Finder behavior
    pseudo = (False,)  # False for original Sound Finder
    estimate = localization.soundfinder(
        reciever_positions,
        arrival_times,
        temperature=20.0,  # celcius
        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
        center=center,  # True for original Sound Finder behavior
        pseudo=pseudo,  # False for original Sound Finder
    )
    assert close(
        np.linalg.norm(np.array(estimate[0:3]) - np.array([10, 10, 0])), 0, 0.1
    )


def test_gillette_localize_2d():
    np.random.seed(0)
    receiver_positions = np.array([[0, 0], [0, 20], [20, 20], [20, 0], [10, 10]])
    sound_source = np.random.rand(2) * 20
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_positions - sound_source, axis=1) / speed_of_sound
    )
    tdoas = time_of_flight - np.min(time_of_flight)

    estimated_pos, _, _ = localization.localize_gillette(receiver_positions, tdoas)

    assert np.allclose(estimated_pos[0:2], sound_source, rtol=0.1)


def test_gillette_localize_3d():
    receiver_positions = np.array(
        [[0, 0, 10], [0, 20, 1], [20, 20, -1], [20, 0, 0.1], [10, 10, 0], [5, 5, 5]]
    )
    sound_source = np.array([10, 12, 2])
    speed_of_sound = 343
    time_of_flight = (
        np.linalg.norm(receiver_positions - sound_source, axis=1) / speed_of_sound
    )
    tdoas = time_of_flight - np.min(time_of_flight)

    estimated_pos, _, _ = localization.localize_gillette(receiver_positions, tdoas)

    assert np.allclose(estimated_pos[0:3], sound_source, atol=2)


def test_localizer():
    aru_coords = pd.read_csv(aru_files, index_col=0)
    preds = pd.read_csv(predictions, index_col=0)
    loca = Localizer(
        files=aru_names,
        predictions=preds_indexed,
        aru_coords=aru_coords,
        sample_rate=32000,
        min_number_of_receivers=2,
        max_distance_between_receivers=100,
        thresholds={"test_species": 0},
        localization_algorithm="soundfinder",
    )
    loca.localize()


assert close(np.mean(loca.locations["predicted_x"], 10))
assert close(np.mean(loca.locations["predicted_y"], 15))
