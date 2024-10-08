#!/usr/bin/env python3
from opensoundscape.audio import Audio
import pytest
import numpy as np
from opensoundscape import signal_processing as sp
import math


@pytest.fixture()
def gpt_wav_str():
    return "tests/audio/great_plains_toad.wav"


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def rugr_wav_str():
    return "tests/audio/rugr_drum.wav"


def test_frequency2scale():
    """test conversion from scale -> frequency -> scale"""
    import pywt

    scale = 5
    sr = 44100
    wavelet = "morl"
    f_hz = pywt.scale2frequency(wavelet, scale) * sr
    assert sp.frequency2scale(f_hz, wavelet, sr) == scale


def test_cwt_peaks(gpt_wav_str):
    a = Audio.from_file(gpt_wav_str, sample_rate=44100).trim(5, 10)
    t, _ = sp.cwt_peaks(a, center_frequency=2500, peak_separation=0.05)
    assert len(t) == 42


def test_find_accel_sequences():
    """search for an accelerating Ruffed Grouse drumming pattern"""
    t = np.array(
        [
            1.02754281,
            1.83007625,
            1.96758198,
            2.17759073,
            3.19263303,
            3.93266386,
            4.52518855,
            5.00270845,
            5.43522647,
            5.80024168,
            5.84524355,
            6.15275636,
            6.44776866,
            6.72278012,
            6.98279095,
            7.20280012,
            7.41030876,
            7.61781741,
            7.79782491,
            7.95533147,
            8.11033793,
            8.26784449,
            8.3928497,
            8.51785491,
            8.64786033,
            8.74786449,
            8.86036918,
            8.95287304,
            9.02787616,
            9.1128797,
            9.25538564,
            9.31788825,
            9.37789075,
            9.66540273,
            9.82290929,
            9.91041293,
            10.0104171,
            10.11792158,
            10.37293221,
            13.94058086,
        ]
    )
    seq_y, seq_t = sp.find_accel_sequences(
        t,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[5, 100],
    )
    assert np.shape(seq_t) == (1, 19)


def test_detect_peak_sequence_cwt(rugr_wav_str):
    """test detection of ruffed grouse drumming

    the default parameters might change, but this should always return
    the same detection.
    """
    rugr_audio = Audio.from_file(rugr_wav_str)
    detections = sp.detect_peak_sequence_cwt(
        rugr_audio,
        sample_rate=400,
        window_len=10,
        center_frequency=50,
        wavelet="morl",
        peak_threshold=0.2,
        peak_separation=15 / 400,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[9, 100],
        plot=False,
    )
    assert len(detections) == 1
    assert detections.iloc[0].seq_len == 24


def test_detect_peak_sequence_cwt_no_results(rugr_wav_str):
    """tests that empty dataframe is returned (instead of errror) if input audio
    is shorter than window_length
    """
    rugr_audio = Audio.from_file(rugr_wav_str).trim(0, 1)
    detections = sp.detect_peak_sequence_cwt(
        rugr_audio,
        sample_rate=400,
        window_len=10,
        center_frequency=50,
        wavelet="morl",
        peak_threshold=0.2,
        peak_separation=15 / 400,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[9, 100],
        plot=False,
    )
    assert len(detections) == 0


def test_detect_peak_sequence_cwt_uneven_length_results(rugr_wav_str):
    """

    this test is for the (resolved) issue #410 in which uneven lengths of
    detected sequences caused a TypeError
    """
    rugr_audio = Audio.from_file(rugr_wav_str).trim(1, 8).loop(length=20)
    detections = sp.detect_peak_sequence_cwt(
        rugr_audio,
        sample_rate=400,
        window_len=3,
        center_frequency=50,
        wavelet="morl",
        peak_threshold=0.2,
        peak_separation=15 / 400,
        dt_range=[0.05, 0.8],
        dy_range=[-0.2, 0],
        d2y_range=[-0.05, 0.15],
        max_skip=3,
        duration_range=[1, 15],
        points_range=[9, 100],
        plot=False,
    )
    assert len(detections) == 2


def test_get_ones_sequence():
    starts, lengths = sp._get_ones_sequences(np.array([1, 1, 1, 0, 1, 1, 0, 1]))
    assert starts == [0, 4, 7]
    assert lengths == [3, 2, 1]
    print(starts, lengths)


def test_thresholded_event_durations():
    starts, lengths = sp.thresholded_event_durations(
        np.array([1, 1, 1, 0, 1, 1, 0, 1]), threshold=0
    )
    assert np.array_equal(starts, np.array([0]))
    assert np.array_equal(lengths, np.array([8]))
    starts, lengths = sp.thresholded_event_durations(
        np.array([1, 1, 1, 0, 1, 1, 0, 1]), threshold=np.array(10)
    )
    assert np.array_equal(starts, np.array([]))
    assert np.array_equal(lengths, np.array([]))
    starts, lengths = sp.thresholded_event_durations(
        np.array([-1, -1, -1, 0, 1, -1, 0, 1]), threshold=-0.1
    )
    assert np.array_equal(starts, np.array([3, 6]))
    assert np.array_equal(lengths, np.array([2, 2]))
    starts, lengths = sp.thresholded_event_durations(
        np.array([]), threshold=np.array(0.99)
    )
    assert np.array_equal(starts, np.array([]))
    assert np.array_equal(lengths, np.array([]))
    starts, lengths = sp.thresholded_event_durations(
        np.array([0, 0, 0]), threshold=np.nan
    )
    assert np.array_equal(starts, np.array([]))
    assert np.array_equal(lengths, np.array([]))
    starts, lengths = sp.thresholded_event_durations(
        np.array([0, np.nan, 0]), threshold=-1
    )
    assert np.array_equal(starts, np.array([0, 2]))
    assert np.array_equal(lengths, np.array([1, 1]))


def test_gcc():
    # test our gcc implementation with an easy case
    np.random.seed(0)
    delay = 200  # samples
    start = 500  # start of signal
    end = 510  # end of signal

    a = np.zeros(1000)
    a[start:end] = 3  # impulse
    a += np.random.rand(1000)  # add noise
    b = np.zeros(1000)
    b[start - delay : end - delay] = (
        3  # signal b is identical to a, but delayed by delay samples
    )
    b += np.random.rand(1000)  # add noise

    for cc_filter in ["cc", "phat", "roth", "scot", "ht"]:
        import scipy

        gccs = sp.gcc(a, b, cc_filter=cc_filter)
        # assert that the argmax is the correct delay
        lags = scipy.signal.correlation_lags(len(a), len(b))
        assert lags[np.argmax(gccs)] == delay


def test_all_tdoa_filter_types_find_correct_delay_no_noise():
    delay = 20  # samples of delay (positive: signal arrives 20 samples later in signal than in reference)
    start = 500  # start of signal
    end = 510  # end of signal

    signal = np.zeros(1000)
    signal[start:end] = 3  # impulse
    reference_signal = np.zeros(1000)
    reference_signal[start - delay : end - delay] = 3

    # add max_delay samples to start and end of reference signal
    max_delay = 50

    for method in ["phat", "roth", "scot", "ht", "cc", "cc_norm"]:
        estimated_sample_delay = sp.tdoa(
            signal,
            reference_signal,
            max_delay=max_delay,
            cc_filter=method,
            sample_rate=1,
        )
        assert estimated_sample_delay == delay


def test_all_tdoa_filter_types_find_correct_delay_with_noise():
    delay = 20  # samples of delay (positive: second signal arrives before first)
    start = 500  # start of signal
    end = 510  # end of signal

    np.random.seed(0)  # is robust to nearly all (but not all) random seeds
    a = np.zeros(1000)
    a[start:end] = 3  # impulse

    a += np.random.rand(1000)  # add noise
    reference_signal = np.zeros(1000)
    reference_signal[start - delay : end - delay] = 3
    reference_signal += np.random.rand(1000)  # add noise

    # add max_delay samples to start and end of reference signal
    max_delay = 50

    for method in ["phat", "roth", "scot", "ht", "cc", "cc_norm"]:
        estimated_sample_delay = sp.tdoa(
            a,
            reference_signal,
            max_delay=max_delay,
            cc_filter=method,
            sample_rate=1,
        )
        assert math.isclose(
            estimated_sample_delay, delay, abs_tol=1
        )  # allow 1 sample error due to float precision

        # # with sample rate !=1
        estimated_delay = sp.tdoa(
            a,
            reference_signal,
            max_delay=max_delay / 22050,
            cc_filter=method,
            sample_rate=22050,
        )
        assert math.isclose(
            estimated_delay, delay / 22050, abs_tol=5e-5
        )  # allow 1 sample error due to float precision


def test_cc_scipy_equivalence():
    ## test our implementation of plain cross-correlation (no filter)
    ## against scipy.signal.correlate, to ensure correctness
    import scipy.signal as sig

    for delay in range(-20, 20):
        start = 500  # start of signal
        end = 510  # end of signal

        a = np.zeros(1000)
        a[start:end] = 3  # impulse
        a += np.random.rand(1000)  # add noise
        b = np.zeros(1000)
        b[start + delay : end + delay] = 3
        b += np.random.rand(1000)
        gccs = sp.gcc(a, b, cc_filter="cc")  # use plain cross-correlation
        # should be exactly the same as scipy.signal.correlate
        np.testing.assert_allclose(gccs, sig.correlate(a, b, mode="full"), 1e-6, 1)


def test_tdoa_return_max():
    # option to return max of cc as well as estimated time delay
    delay = 20  # samples of delay (positive: second signal arrives before first)
    start = 500  # start of signal
    end = 510  # end of signal

    a = np.zeros(1000)
    a[start:end] = 3  # impulse
    reference_signal = np.zeros(1000)
    reference_signal[start - delay : end - delay] = 3

    # set max_delay
    max_delay = 50

    # filter methods will change the output values of cc, but plain cc gives expected value
    delay, cc_max = sp.tdoa(
        a, reference_signal, max_delay, cc_filter="cc", sample_rate=1, return_max=True
    )
    assert math.isclose(cc_max, 3 * 3 * (end - start), abs_tol=1e-4)


def test_tdoa_max_delay_true_delay_higher():
    # test if max_delay works in limiting the search space
    # by setting the true delay to be higher than max_delay
    max_delay = 19  # samples
    delay = 100  # samples of delay (positive: second signal arrives before first)
    start = 500  # start of signal
    end = 510  # end of signal

    a = np.zeros(1000)
    a[start:end] = 3  # impulse
    a += np.random.rand(1000)  # add noise
    b = np.zeros(1000)
    b[start - delay : end - delay] = 3
    b += np.random.rand(1000)
    delay, cc_max = sp.tdoa(
        a, b, cc_filter="cc", sample_rate=1, return_max=True, max_delay=max_delay
    )
    assert abs(delay) <= max_delay


def test_tdoa_max_delay_true_delay_within():
    # test if max_delay works in limiting the search space
    # this time, ensure that the true delay is within the search space
    delay = 49  # samples of delay (positive: second signal arrives before first)
    start = 500  # start of signal
    end = 510  # end of signal

    a = np.zeros(1000)
    a[start:end] = 3  # impulse
    a += np.random.rand(1000)  # add noise

    reference_signal = np.zeros(1000)
    reference_signal[start - delay : end - delay] = 3
    reference_signal += np.random.rand(1000)

    # add max_delay samples to start and end of reference signal
    max_delay = 50

    delay, cc_max = sp.tdoa(
        a,
        reference_signal,
        cc_filter="cc",
        sample_rate=1,
        return_max=True,
        max_delay=max_delay,
    )
    assert delay == 49
