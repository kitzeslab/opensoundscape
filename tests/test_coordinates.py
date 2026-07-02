"""Tests for coordinate reference system (CRS) support in localization.

These tests require the optional dependency `pyproj`. If it is not installed, the whole
module is skipped rather than failing.
"""

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyproj")

from opensoundscape import localization
from opensoundscape.localization import coordinates


def test_utm_epsg_for_lonlat():
    # Pittsburgh, PA (northern hemisphere, UTM zone 17N -> EPSG:32617)
    assert coordinates.utm_epsg_for_lonlat(-79.96, 40.44) == 32617
    # southern hemisphere point (Sydney) -> 327xx
    assert coordinates.utm_epsg_for_lonlat(151.21, -33.87) == 32756


def test_lonlat_xy_roundtrip():
    lon, lat = -79.96, 40.44
    x, y, crs = coordinates.lonlat_to_xy(lon, lat)
    # auto-selected UTM zone for this point
    assert crs == 32617
    lon2, lat2 = coordinates.xy_to_lonlat(x, y, crs)
    assert math.isclose(lon, float(lon2), abs_tol=1e-6)
    assert math.isclose(lat, float(lat2), abs_tol=1e-6)


def test_lonlat_xy_roundtrip_array():
    lons = np.array([-79.960, -79.961, -79.959])
    lats = np.array([40.440, 40.441, 40.439])
    xs, ys, crs = coordinates.lonlat_to_xy(lons, lats)
    back_lon, back_lat = coordinates.xy_to_lonlat(xs, ys, crs)
    assert np.allclose(lons, back_lon, atol=1e-6)
    assert np.allclose(lats, back_lat, atol=1e-6)


def test_lonlat_to_xy_explicit_crs():
    # passing an explicit crs should use it rather than auto-selecting
    _, _, crs = coordinates.lonlat_to_xy(-79.96, 40.44, crs="EPSG:32617")
    assert crs == "EPSG:32617"


def test_xy_to_lonlat_requires_crs():
    with pytest.raises(ValueError):
        coordinates.xy_to_lonlat(1.0, 2.0, crs=None)


def test_project_file_coords_2d_and_3d():
    df = pd.DataFrame(
        {
            "longitude": [-79.960, -79.961, -79.959],
            "latitude": [40.440, 40.441, 40.439],
        },
        index=["a.wav", "b.wav", "c.wav"],
    )
    projected, crs = coordinates.project_file_coords(df)
    assert crs == 32617
    assert list(projected.columns) == ["x", "y"]
    assert list(projected.index) == ["a.wav", "b.wav", "c.wav"]
    assert "z" not in projected.columns

    # elevation column is carried through as z
    df["elevation"] = [100.0, 101.0, 102.0]
    projected_3d, _ = coordinates.project_file_coords(df)
    assert list(projected_3d.columns) == ["x", "y", "z"]
    assert list(projected_3d["z"]) == [100.0, 101.0, 102.0]


def test_project_file_coords_missing_column_raises():
    df = pd.DataFrame({"lon": [-79.96], "lat": [40.44]}, index=["a.wav"])
    with pytest.raises(ValueError):
        coordinates.project_file_coords(df)  # default column names not present


def test_from_lonlat_sets_crs_and_projects():
    df = pd.DataFrame(
        {
            "longitude": [-79.960, -79.961, -79.959, -79.962],
            "latitude": [40.440, 40.441, 40.439, 40.442],
        },
        index=["a.wav", "b.wav", "c.wav", "d.wav"],
    )
    array = localization.SynchronizedRecorderArray.from_lonlat(df)
    assert array.crs == 32617
    assert list(array.file_coords.columns) == ["x", "y"]
    assert len(array.file_coords) == 4


def test_position_estimate_lonlat_roundtrip():
    lon, lat = -79.96, 40.44
    x, y, crs = coordinates.lonlat_to_xy(lon, lat)
    pos = localization.PositionEstimate(location_estimate=np.array([x, y]), crs=crs)
    recovered = pos.location_estimate_lonlat
    assert math.isclose(recovered[0], lon, abs_tol=1e-6)
    assert math.isclose(recovered[1], lat, abs_tol=1e-6)


def test_position_estimate_lonlat_requires_crs():
    pos = localization.PositionEstimate(location_estimate=np.array([100.0, 200.0]))
    with pytest.raises(ValueError):
        pos.location_estimate_lonlat


def test_position_estimate_lonlat_none_location():
    pos = localization.PositionEstimate(location_estimate=None, crs=32617)
    assert pos.location_estimate_lonlat is None


def test_position_estimate_crs_survives_dict_roundtrip():
    pos = localization.PositionEstimate(
        location_estimate=np.array([1.0, 2.0]), crs=32617
    )
    recovered = localization.PositionEstimate.from_dict(pos.to_dict())
    assert recovered.crs == 32617
