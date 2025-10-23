"""Tests for MSRP localization functions (updated API)

These tests verify the updated API surface for MSRP functions:
- SearchMap class creation and time-interval computation
- compute_msrp(signals, search_map, freq_low, freq_high, cc_filter, aggregation_fn)
  returns a pd.Series indexed by grid-point tuples
- localize(signals, search_map, ...) returns a dict with 'location' and 'max_power'
  and includes 'power_map' and 'search_map' when keep_maps=True

The tests use small synthetic signals so they run quickly and focus on API correctness
and basic integration rather than high-precision localization.
"""

import numpy as np
import pandas as pd
import pytest

from opensoundscape.localization import msrp


@pytest.fixture()
def simple_receiver_positions_df():
    """Return a small DataFrame of 4 receiver positions (2D) with receiver IDs as index."""
    coords = pd.DataFrame(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        columns=["x", "y"],
        index=["r0", "r1", "r2", "r3"],
    )
    return coords


@pytest.fixture()
def tiny_signals():
    """Create tiny synchronized signals for 4 receivers (dict of numpy arrays)

    Signals are short (256 samples) random noise; these tests only check API
    behavior not accurate localization.
    """
    rng = np.random.RandomState(0)
    sig_len = 256
    signals = {f"r{i}": rng.normal(0, 1, sig_len) for i in range(4)}
    return signals, 8000


@pytest.fixture()
def synthetic_3d():
    """Create 3D receiver positions, synthetic signals, and true source for tests."""
    rec_pos = pd.DataFrame(
        {
            "x": [0.0, 8.0, 15.0, -5.0],
            "y": [0.0, -2.0, 10.0, 12.0],
            "z": [0.0, 1.0, 2.0, 3.0],
        },
        index=["r0", "r1", "r2", "r3"],
    )

    source = np.array([3.5, 7.2, 1.1])
    speed_of_sound = 343.0
    sample_rate = 16000
    duration = 0.2
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate

    freq = 2000.0
    rng = np.random.RandomState(0)
    signals = {}
    for rid, row in rec_pos.iterrows():
        dist = np.linalg.norm(row.values - source)
        delay = dist / speed_of_sound
        center = delay + 0.05
        pulse = np.exp(-((t - center) ** 2) / (2 * (0.0015**2))) * np.sin(
            2 * np.pi * freq * (t - center)
        )
        pulse += rng.normal(0, 0.001, size=pulse.shape)
        signals[rid] = pulse

    return signals, rec_pos, source, sample_rate


@pytest.fixture()
def synthetic_2d():
    """Create 2D receiver positions (x,y), synthetic signals, and true source for tests."""
    rec_pos = pd.DataFrame(
        [[0.0, 0.0], [12.0, -3.0], [18.0, 9.0], [-6.0, 14.0]],
        columns=["x", "y"],
        index=["r0", "r1", "r2", "r3"],
    )

    source = np.array([4.3, 5.7])
    speed_of_sound = 343.0
    sample_rate = 16000
    duration = 0.2
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate

    freq = 1600.0
    rng = np.random.RandomState(1)
    signals = {}
    for rid, row in rec_pos.iterrows():
        dist = np.linalg.norm(row.values - source)
        delay = dist / speed_of_sound
        center = delay + 0.05
        pulse = np.exp(-((t - center) ** 2) / (2 * (0.0015**2))) * np.sin(
            2 * np.pi * freq * (t - center)
        )
        pulse += rng.normal(0, 0.001, size=pulse.shape)
        signals[rid] = pulse

    return signals, rec_pos, source, sample_rate


def test_searchmap_and_time_intervals(simple_receiver_positions_df):
    """Test SearchMap creation and time-interval precomputation (no errors)."""
    receiver_positions = simple_receiver_positions_df
    sample_rate = 8000

    # create a SearchMap with 1m resolution and margin 0
    smap = msrp.SearchMap(
        receiver_positions=receiver_positions,
        sample_rate=sample_rate,
        resolution=5.0,
        margin=0,
        speed_of_sound=343,
        compute_time_intervals=True,
    )

    # SearchMap should have search_points DataFrame and time_delay_min/max DataFrames
    assert hasattr(smap, "search_points")
    assert hasattr(smap, "time_delay_min") and hasattr(smap, "time_delay_max")

    # time_delay DataFrames use coordinate tuples as the index; verify they
    # correspond to the search_points coordinates
    search_point_tuples = [tuple(row) for row in smap.search_points.values]
    assert list(smap.time_delay_min.index) == search_point_tuples
    assert list(smap.time_delay_max.index) == search_point_tuples


def test_compute_msrp_returns_series(simple_receiver_positions_df, tiny_signals):
    """Test compute_msrp signature and return type."""
    receiver_positions = simple_receiver_positions_df
    signals, sr = tiny_signals

    smap = msrp.SearchMap(
        receiver_positions=receiver_positions,
        sample_rate=sr,
        resolution=5.0,
        margin=0,
        compute_time_intervals=True,
    )

    # compute SRP (we don't expect a specific localization here)
    power_map = msrp.compute_msrp(
        signals=signals,
        search_map=smap,
        freq_low=None,
        freq_high=None,
        cc_filter="phat",
        aggregation_fn=np.sum,
    )

    # Should return a pandas Series indexed by grid-point coordinate tuples
    assert isinstance(power_map, pd.Series)
    assert all(isinstance(idx, tuple) for idx in power_map.index)
    assert power_map.size == len(smap.search_points)


def test_localize_api_and_keep_maps(simple_receiver_positions_df, tiny_signals):
    """Test localize() returns expected dict keys and attaches maps when requested."""
    receiver_positions = simple_receiver_positions_df
    signals, sr = tiny_signals

    smap = msrp.SearchMap(
        receiver_positions=receiver_positions,
        sample_rate=sr,
        resolution=5.0,
        margin=0,
        compute_time_intervals=True,
    )

    out = msrp.localize(
        signals=signals,
        search_map=smap,
        freq_low=None,
        freq_high=None,
        keep_maps=True,
        cc_filter="phat",
        aggregation_fn=np.sum,
        convex_hull_margin=0,
        detrend=True,
    )

    # Basic contract: must contain 'location' and 'max_power'
    assert isinstance(out, dict)
    assert "location" in out and "max_power" in out

    # When keep_maps=True we also expect 'power_map' and 'search_map'
    assert "power_map" in out and "search_map" in out

    # location should be one of the search point coordinate tuples
    assert out["location"] in list(smap.time_delay_min.index)


def test_localize_invalid_inputs_raises(simple_receiver_positions_df, tiny_signals):
    """Test that localize validates required input types (basic checks)."""
    receiver_positions = simple_receiver_positions_df
    signals, sr = tiny_signals

    smap = msrp.SearchMap(
        receiver_positions=receiver_positions,
        sample_rate=sr,
        resolution=5.0,
        margin=0,
        compute_time_intervals=True,
    )

    # signals must be a dict
    with pytest.raises(Exception):
        msrp.localize(
            signals=[1, 2, 3],
            search_map=smap,
            freq_low=None,
            freq_high=None,
        )

    # search_map must be a SearchMap-like object
    with pytest.raises(Exception):
        msrp.localize(
            signals=signals,
            search_map=object(),
            freq_low=None,
            freq_high=None,
        )


def test_localize_synthetic_recovers_position():
    """Synthetic 3D test: non-square grid and non-equal source coords

    Create four receivers on an irregular (non-square) layout with z values,
    synthesize short time-delayed pulses at each receiver for a source at a
    non-equal (x,y,z) location, run `msrp.localize` and assert the estimated
    location is within a small radius of the true source.
    """
    # receiver positions (non-square, 3D)
    rec_pos = pd.DataFrame(
        {
            "x": [0.0, 8.0, 15.0, -5.0],
            "y": [0.0, -2.0, 10.0, 12.0],
            "z": [0.0, 1.0, 2.0, 3.0],
        },
        index=["r0", "r1", "r2", "r3"],
    )

    # true source (non-equal coordinates)
    source = np.array([3.5, 7.2, 1.1])

    speed_of_sound = 343.0
    sample_rate = 16000
    duration = 0.2
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate

    # synthesize a Gaussian-modulated tone arriving at appropriate delays
    freq = 2000.0
    signals = {}
    for rid, row in rec_pos.iterrows():
        dist = np.linalg.norm(row.values - source)
        delay = dist / speed_of_sound
        # center the pulse at delay + 0.05s to avoid 0 index
        center = delay + 0.05
        pulse = np.exp(-((t - center) ** 2) / (2 * (0.0015**2))) * np.sin(
            2 * np.pi * freq * (t - center)
        )
        # small noise
        pulse += np.random.RandomState(0).normal(0, 0.001, size=pulse.shape)
        signals[rid] = pulse

    # create search map with moderate resolution (non-square extent will be implicit)
    smap = msrp.SearchMap(
        receiver_positions=rec_pos,
        sample_rate=sample_rate,
        resolution=0.5,
        margin=5.0,
        compute_time_intervals=True,
    )

    out = msrp.localize(
        signals=signals,
        search_map=smap,
        freq_low=500,
        freq_high=4000,
        keep_maps=True,
        cc_filter="phat",
        aggregation_fn=np.sum,
        convex_hull_margin=0,
        detrend=True,
    )

    estimated = np.array(out["location"])

    # Allow a tolerance that accounts for grid discretization (1m resolution)
    tol = 1.6
    error = np.linalg.norm(estimated - source)
    assert error <= tol, f"Localization error {error:.2f}m exceeds tolerance {tol}m"


def test_localize_synthetic_2d_recovers_position(synthetic_2d):
    """Test 2D localization recovers XY location within tolerance."""
    signals, rec_pos, source, sample_rate = synthetic_2d

    smap = msrp.SearchMap(
        receiver_positions=rec_pos,
        sample_rate=sample_rate,
        resolution=0.5,
        margin=5.0,
        compute_time_intervals=True,
    )

    out = msrp.localize(
        signals=signals,
        search_map=smap,
        freq_low=400,
        freq_high=3000,
        keep_maps=True,
        cc_filter="phat",
        aggregation_fn=np.sum,
        convex_hull_margin=0,
        detrend=True,
    )

    estimated = np.array(out["location"])

    # only compare x,y for 2D
    est_xy = estimated[:2]
    error = np.linalg.norm(est_xy - source)
    tol = 1.6
    assert error <= tol, f"2D localization error {error:.2f}m exceeds tolerance {tol}m"


if __name__ == "__main__":
    pytest.main([__file__])
