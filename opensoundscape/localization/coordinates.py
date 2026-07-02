"""Coordinate reference system (CRS) helpers for acoustic localization.

The localization algorithms in OpenSoundscape operate on Cartesian coordinates in
meters (e.g. a UTM coordinate system). Field recorders, however, typically log their
positions as geographic coordinates (longitude/latitude, WGS84). This module provides
lightweight helpers to project geographic coordinates to a metric CRS before
localization, and to convert estimated source locations back to longitude/latitude.

These helpers depend on ``pyproj``, which is an optional dependency. Install it with::

    pip install opensoundscape[localization]

or::

    pip install pyproj
"""

import numpy as np
import pandas as pd

# WGS84 geographic coordinate system (longitude/latitude in degrees)
WGS84 = "EPSG:4326"


def _require_pyproj():
    """Import pyproj, raising a helpful error if it is not installed.

    Returns:
        the imported pyproj module
    """
    try:
        import pyproj
    except ImportError as exc:  # pragma: no cover - exercised only without pyproj
        raise ImportError(
            "This feature requires the optional dependency `pyproj`, which was not "
            "found in your environment. Install it with `pip install pyproj` or "
            "`pip install opensoundscape[localization]`."
        ) from exc
    return pyproj


def utm_epsg_for_lonlat(longitude, latitude):
    """Return the EPSG code of the UTM zone containing a longitude/latitude point.

    Args:
        longitude: longitude in decimal degrees (WGS84)
        latitude: latitude in decimal degrees (WGS84)

    Returns:
        int EPSG code for the corresponding WGS84 / UTM zone (326xx for the northern
        hemisphere, 327xx for the southern hemisphere)
    """
    # UTM zones are 6 degrees wide, numbered 1-60 starting at longitude -180
    zone = int((longitude + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    if latitude >= 0:
        return 32600 + zone  # northern hemisphere
    return 32700 + zone  # southern hemisphere


def lonlat_to_xy(longitude, latitude, crs=None):
    """Project longitude/latitude (WGS84) coordinates to a metric CRS, in meters.

    Args:
        longitude: scalar or array of longitudes in decimal degrees (WGS84)
        latitude: scalar or array of latitudes in decimal degrees (WGS84)
        crs: target coordinate reference system for the projected coordinates. Anything
            accepted by ``pyproj`` (e.g. an int EPSG code, "EPSG:32617", a
            ``pyproj.CRS``). If None (default), an appropriate WGS84 / UTM zone is
            chosen automatically from the mean of the input coordinates.

    Returns:
        (x, y, crs) tuple where x and y are the projected coordinates in meters (matching
        the shape of the inputs) and crs is the CRS that was used (useful when crs was
        chosen automatically, so it can be re-used for the inverse transform)
    """
    pyproj = _require_pyproj()
    longitude = np.asarray(longitude, dtype="float64")
    latitude = np.asarray(latitude, dtype="float64")

    if crs is None:
        crs = utm_epsg_for_lonlat(float(longitude.mean()), float(latitude.mean()))

    # always_xy=True so inputs/outputs are ordered (longitude, latitude) / (x, y)
    transformer = pyproj.Transformer.from_crs(WGS84, crs, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    return x, y, crs


def xy_to_lonlat(x, y, crs):
    """Convert projected coordinates (in meters) back to longitude/latitude (WGS84).

    Args:
        x: scalar or array of x (easting) coordinates in meters
        y: scalar or array of y (northing) coordinates in meters
        crs: coordinate reference system of the (x, y) coordinates. Anything accepted by
            ``pyproj`` (e.g. an int EPSG code, "EPSG:32617", a ``pyproj.CRS``).

    Returns:
        (longitude, latitude) tuple in decimal degrees (WGS84), matching the shape of
        the inputs
    """
    if crs is None:
        raise ValueError(
            "crs must be provided to convert projected coordinates to longitude/latitude"
        )
    pyproj = _require_pyproj()
    transformer = pyproj.Transformer.from_crs(crs, WGS84, always_xy=True)
    longitude, latitude = transformer.transform(np.asarray(x), np.asarray(y))
    return longitude, latitude


def project_file_coords(
    file_coords,
    crs=None,
    longitude_column="longitude",
    latitude_column="latitude",
    elevation_column="elevation",
):
    """Project a table of receiver longitude/latitude coordinates to meters.

    Converts a DataFrame of receiver positions given as longitude/latitude (WGS84) into
    the (x, y) or (x, y, z) meter coordinates expected by
    :class:`~opensoundscape.localization.synchronized_recorder_array.SynchronizedRecorderArray`.

    Args:
        file_coords: pandas.DataFrame indexed by audio file path, with columns for
            longitude and latitude (and optionally elevation). Column names are
            configurable with the arguments below.
        crs: target metric CRS for the projected coordinates (see :func:`lonlat_to_xy`).
            If None (default), an appropriate WGS84 / UTM zone is chosen automatically.
        longitude_column: name of the longitude column [default: "longitude"]
        latitude_column: name of the latitude column [default: "latitude"]
        elevation_column: name of an optional elevation/height column in meters. If this
            column is present it is carried through unchanged as the z coordinate
            [default: "elevation"]

    Returns:
        (projected, crs) tuple where projected is a pandas.DataFrame with the same index
        and columns "x", "y" (and "z" if an elevation column was present), in meters, and
        crs is the CRS that was used for the projection
    """
    for col in (longitude_column, latitude_column):
        if col not in file_coords.columns:
            raise ValueError(
                f"file_coords is missing the column '{col}'. Columns present: "
                f"{list(file_coords.columns)}. Pass longitude_column/latitude_column "
                "to specify the correct column names."
            )

    x, y, crs = lonlat_to_xy(
        file_coords[longitude_column].values,
        file_coords[latitude_column].values,
        crs=crs,
    )
    projected = pd.DataFrame({"x": x, "y": y}, index=file_coords.index)
    if elevation_column in file_coords.columns:
        projected["z"] = file_coords[elevation_column].values
    return projected, crs
