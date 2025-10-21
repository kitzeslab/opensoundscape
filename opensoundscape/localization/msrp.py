"""MSRP (Maximum Source Range Power) localization algorithm

This module contains Python translations of the MSRP localization functions
originally implemented in Matlab by Tim Huang, then in R by Richard Hedley (locaR package)

#TODO: refactor to use list of (x,y,z) positions instead of requiring cubic grid
"""

import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal
from opensoundscape.signal_processing import gcc


def makeSearchMap(
    easting, northing, elevation, margin=10, zMin=-1, zMax=10, resolution=1
):
    """Create a 3D search grid for sound source localization.

    Creates a three-dimensional array over which to search for sound sources.
    The output arrays have shape (len(XAxis), len(YAxis), len(ZAxis)) for
    intuitive (x, y, z) indexing.

    Args:
        easting: array-like, x coordinates of nodes in meters
        northing: array-like, y coordinates of nodes in meters
        elevation: array-like, z coordinates of nodes in meters
        margin: float, margin in meters around nodes to search (default: 10)
        zMin: float, minimum elevation offset from lowest node in meters (default: -1)
        zMax: float, maximum elevation offset from highest node in meters (default: 10)
        resolution: float, grid resolution in meters (default: 1)

    Returns:
        dict containing:
            - 'XAxis', 'YAxis', 'ZAxis': 1D arrays with coordinate values
            - 'XDen', 'YDen', 'ZDen': grid resolution (all equal to resolution)
            - 'XLim', 'YLim', 'ZLim': tuples with (min, max) for each dimension

    Example:
        >>> coords = pd.DataFrame({
        ...     'Station': ['A', 'B', 'C'],
        ...     'Easting': [0, 100, 50],
        ...     'Northing': [0, 0, 100],
        ...     'Elevation': [0, 0, 0]
        ... })
        >>> sm = makeSearchMap(
        ...     easting=coords['Easting'],
        ...     northing=coords['Northing'],
        ...     elevation=coords['Elevation'],
        ...     margin=10,
        ...     zMin=-1,
        ...     zMax=10,
        ...     resolution=1
        ... )
    """
    easting = np.asarray(easting)
    northing = np.asarray(northing)
    elevation = np.asarray(elevation)

    # Calculate grid limits
    XLim1 = np.min(easting) - margin
    XLim2 = np.max(easting) + margin
    YLim1 = np.min(northing) - margin
    YLim2 = np.max(northing) + margin
    ZLim1 = np.min(elevation) + zMin
    ZLim2 = np.max(elevation) + zMax

    # Create axes
    XAxis = np.arange(XLim1, XLim2 + resolution, resolution)
    YAxis = np.arange(YLim1, YLim2 + resolution, resolution)
    ZAxis = np.arange(ZLim1, ZLim2 + resolution, resolution)

    SearchMap = {
        "XAxis": XAxis,
        "YAxis": YAxis,
        "ZAxis": ZAxis,
        "resolution": resolution,
        "XLim": (XLim1, XLim2),
        "YLim": (YLim1, YLim2),
        "ZLim": (ZLim1, ZLim2),
    }

    return SearchMap


def compute_time_delays(
    receiver_positions, search_map, sample_rate, speed_of_sound, signal_len
):
    """Pre-compute time-delays at search positions

    Args:
        receiver_positions: ndarray of shape (N, 3) containing [x, y, z] positions of N receivers
        (microphones) search_map: dict with keys 'resolution' and
            'XAxis', 'YAxis', 'ZAxis' (1D arrays of search grid coordinates)
        sample_rate: float, sample rate in Hz speed_of_sound: float, speed of sound in m/s data_len:
        int, length of audio data in samples

    Returns:
        dict containing InitData with precomputed time delay indices as 3D arrays. The 'T1' and 'T2'
        arrays have shape (nx, ny, nz, NIJ) where:
            - nx, ny, nz are the spatial dimensions from search_map
            - NIJ is the number of receiver pairs

    Author: Sam Lapp, translated from Tim Huang (original Matlab implementation) and Richard Hedley
    (R implementation)
    """
    import time

    start_time = time.time()

    receiver_positions = np.asarray(receiver_positions)
    n_receivers = len(receiver_positions)  # Number of receivers

    sr = sample_rate  # for brevity

    cc_len = signal_len * 2 - 1  # Length of cross-correlation result

    # Number of pairwise comparisons of receivers
    n_receiver_pairs = n_receivers * (n_receivers - 1) // 2

    # Create list of all pairwise comparisons
    import itertools

    pairs = np.array(list(itertools.combinations(range(n_receivers), 2)), dtype=int).T
    # rec1, rec2 = receiver_pairs[:, 0], receiver_pairs[:, 1]

    # Get spatial dimensions from search_map
    nx, ny, nz = (
        len(search_map["XAxis"]),
        len(search_map["YAxis"]),
        len(search_map["ZAxis"]),
    )

    # Preallocate 3D arrays for pairwise time-difference-of-arrival for
    # each search grid location and pair of receivers
    # Shape: (nx, ny, nz, n_receiver_pairs) for each grid point and receiver pair
    T1 = np.zeros((nx, ny, nz, n_receiver_pairs), dtype=int)
    T2 = np.zeros((nx, ny, nz, n_receiver_pairs), dtype=int)

    # Iterate over all grid points using spatial indices
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Get coordinates of this grid point
                x = search_map["XAxis"][ix]
                y = search_map["YAxis"][iy]
                z = search_map["ZAxis"][iz]

                # Distance vector from location (x,y,z) to each of N receivers
                # Add small epsilon to avoid zero distances
                rel_receiver_position = np.array([x, y, z]) - receiver_positions
                receiver_dist = np.sqrt(np.sum(rel_receiver_position**2, axis=1)) + 1e-6

                # Normalized directional differences divided by speed_of_sound (gradient of time delay)
                # eq 9
                grad_tau_x = (
                    (x - receiver_positions[pairs[1], 0]) / receiver_dist[pairs[1]]
                    - (x - receiver_positions[pairs[0], 0]) / receiver_dist[pairs[0]]
                ) / speed_of_sound
                grad_tau_y = (
                    (y - receiver_positions[pairs[1], 1]) / receiver_dist[pairs[1]]
                    - (y - receiver_positions[pairs[0], 1]) / receiver_dist[pairs[0]]
                ) / speed_of_sound
                grad_tau_z = (
                    (z - receiver_positions[pairs[1], 2]) / receiver_dist[pairs[1]]
                    - (z - receiver_positions[pairs[0], 2]) / receiver_dist[pairs[0]]
                ) / speed_of_sound

                # magnitude of the gradient
                grad_mag = np.sqrt(grad_tau_x**2 + grad_tau_y**2 + grad_tau_z**2)

                """
                Quoting the paper:

                The IMTDF inside a volume can only take values in the range defined by its boundary
                surface. Therefore, for each point of the grid, the problem of finding the GCC
                accumulation limits of its volume of influence can be simplified to finding the
                maximum and minimum values on the boundary surface.
                ...
                The accumulation limits for a symmetric volume surrounding a point of the grid can
                be calculated by taking the product of the magnitude of the gradient and the
                distance d that exists from the point to the boundary following the gradient’s
                direction
                """
                # Calculate distance to boundary d following the gradient direction
                theta = np.arccos(grad_tau_z / grad_mag)  # eq 13
                phi = np.arctan2(grad_tau_x, grad_tau_y)  # eq 14
                angles = np.vstack(
                    [
                        np.cos(phi) * np.sin(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(theta),
                    ]
                )
                # eq 12
                d = 0.5 * search_map["resolution"] * np.min(1 / np.abs(angles), axis=0)

                # Compute time delays in seconds
                paired_distance_diff = receiver_dist[pairs[1]] - receiver_dist[pairs[0]]
                dT1 = (paired_distance_diff) / speed_of_sound - grad_mag * d
                dT2 = (paired_distance_diff) / speed_of_sound + grad_mag * d

                # convert to sample position in gcc result
                # Add signal_len-1 to get 0-based center index
                dT1 = np.round(dT1 * sr) + (signal_len - 1)
                dT2 = np.round(dT2 * sr) + (signal_len - 1)

                # Clamp to valid indices [0, ML2-1]
                dT1 = np.clip(dT1, 0, cc_len - 1)
                dT2 = np.clip(dT2, 0, cc_len - 1)

                # Store in 3D arrays
                T1[ix, iy, iz, :] = dT1.astype(int)  # .flatten()
                T2[ix, iy, iz, :] = dT2.astype(int)  # .flatten()

    elapsed = time.time() - start_time
    print(f"Computed pairwise arrival times in {elapsed:.1f} seconds")

    pairwise_arrival_times = {"T1": T1, "T2": T2}

    return pairwise_arrival_times


def compute_msrp(
    receiver_positions,
    signals,
    sample_rate,
    freq_low,
    freq_high,
    time_delays,
    cc_filter="phat",
):
    """Calculate likelihood of sound sources at each search location.

    This function uses the InitData and other info to calculate the likelihood
    of sound sources coming from each location using generalized cross-correlation
    with the Hilbert transform.

    Uses "level 2" implementation of original Matlab implementation

    Args:
        receiver_positions: ndarray of shape (N, 3) containing [x, y, z] positions
        signals: ndarray of shape (N, DataLen) containing audio samples from each node
        sample_rate: float, sample rate in Hz
        freq_low: float, low frequency cutoff in Hz
        freq_high: float, high frequency cutoff in Hz
        time_delays: dict created with compute_time_delays function, the valid region of
            time delay indices for each search location
        cc_filter: str, cross-correlation filter to use; see opensoundscape.signal_processing.gcc
            (options: phat, roth, scot, ht, cc, cc_norm)

    Returns:
        ndarray: SMap array with likelihood values for each search location,
            shape (nx, ny, nz) matching the search_map dimensions

    Author: Sam Lapp, translated from Tim Huang (original Matlab implementation) and Richard Hedley
    (R implementation)
    """
    receiver_positions = np.asarray(receiver_positions)
    n_receivers = len(receiver_positions)  # Number of receivers
    cc_len = signals.shape[1] * 2 - 1  # Length of cross-correlation result
    n_receiver_pairs = n_receivers * (n_receivers - 1) // 2  # Number of receiver pairs

    k = 0
    pairwise_cc = np.zeros((n_receiver_pairs, cc_len))  # Preallocate cross-correlation

    # Compute generalized cross-correlation for all pairs
    for i in range(n_receivers - 1):
        for j in range(i + 1, n_receivers):
            # Use the gcc function from signal_processing.py
            # with frequency range filtering; note ordering convention
            gcc_result = gcc(
                signals[j, :],  # signal
                signals[i, :],  # reference signal
                cc_filter=cc_filter,  # phat is used in original implementation
                frequency_range=(freq_low, freq_high),
                sample_rate=sample_rate,
            )

            # Apply Hilbert transform and take absolute value
            # Note: scipy.signal.hilbert returns the analytic signal
            analytic_signal = scipy_signal.hilbert(gcc_result)
            pairwise_cc[k, :] = np.abs(analytic_signal)
            k += 1

    # Get spatial dimensions and time delay arrays from time_delays
    # T1 = time_delays["T1"]  # Shape: (nx, ny, nz, n_receiver_pairs)
    # T2 = time_delays["T2"]  # Shape: (nx, ny, nz, n_receiver_pairs)
    nx, ny, nz, _ = time_delays["T1"].shape

    # Initialize steered response power array with spatial dimensions
    srp = np.zeros((nx, ny, nz))

    # Calculate steered response power for each spatial location
    # by integrating over valid time delay ranges
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):

                sub1 = []
                sub2 = []
                for kij in range(n_receiver_pairs):
                    # Build indices for valid time delay range
                    # They should already be clamped to [0, ML2-1]
                    t1 = time_delays["T1"][ix, iy, iz, kij]
                    t2 = time_delays["T2"][ix, iy, iz, kij]
                    if t2 >= t1:  # >0 interval to sum over
                        n_indices = t2 - t1 + 1
                        sub1.extend([kij] * n_indices)
                        sub2.extend(range(t1, t2 + 1))

                # Extract relevant correlation values and sum
                # extensions of the approach use different aggregation
                if len(sub1) > 0:
                    tR = pairwise_cc[sub1, sub2]
                    srp[ix, iy, iz] = np.sum(tR)
                else:
                    srp[ix, iy, iz] = 0

    return srp


def localize(
    wav_dict,
    coordinates,
    sample_rate,
    search_map,
    time_delays,
    freq_low=2000,
    freq_high=8000,
    plot=False,
    keep_power_map=True,
):
    """Localize a sound source using MSRP algorithm.

    Main function to localize a sound source given synchronized audio recordings from multiple
    stations using the MSRP (Modified Steered Response Power) algorithm.

    Args:
        wav_dict: dict of audio arrays where keys are station names and values
            are 1D numpy arrays of audio samples. All must have same sample rate and length.
        coordinates: pandas DataFrame with columns:
            - 'Station': station name (must match keys in wav_dict)
            - 'Easting': x coordinate in meters
            - 'Northing': y coordinate in meters
            - 'Elevation': z coordinate in meters
        sample_rate: float, sample rate of audio in Hz margin: float, margin in meters around
            stations to search (default: 10) zMin: float, minimum elevation offset from lowest station
            in meters (default: -1) zMax: float, maximum elevation offset from highest station in meters
            (default: 20)
        resolution: float, grid resolution in meters (default: 1) freq_low: float, low
            frequency cutoff in Hz (default: 2000)
        freq_high: float, high frequency cutoff in Hz
            (default: 8000)
        temp_c: float, temperature in Celsius for calculating speed of sound
            (default: 15)
        speed_of_sound: float, speed of sound in m/s. If provided, overrides temp_c
            (default: None)
        plot: bool, whether to create plots (default: False) loc_folder: str, folder
            path for saving plots (required if plot=True)
        time_delays: dict, precomputed pairwise time delays from compute_time_delays
        keep_power_map: bool, whether to return the power map in output (default: True)

    Returns:
        dict containing:
            - 'location': DataFrame with Easting, Northing, Elevation, Power
            - 'smap': (if keep_search_map=True) 3D likelihood array

    Author: Sam Lapp (Python implementation); Richard Hedley (R implementation); Tim Huang (original
    Matlab implementation);
    """
    import time

    # Check that wav_dict is a dict with named entries
    if not isinstance(wav_dict, dict):
        raise ValueError("wav_dict must be a dictionary with station names as keys")

    # Check that all station names in wav_dict are in coordinates
    if not all(name in coordinates["Station"].values for name in wav_dict.keys()):
        raise ValueError("Some names in wav_dict not found in coordinates!")

    # Get station names
    stations = list(wav_dict.keys())

    # Create node position array from coordinates
    coords_indexed = coordinates.set_index("Station")
    receiver_positions = coords_indexed.loc[
        stations, ["Easting", "Northing", "Elevation"]
    ].values

    first_audio = list(wav_dict.values())[0]
    data_len = len(first_audio)

    # Verify all audio arrays have the same length
    for name, audio in wav_dict.items():
        if len(audio) != data_len:
            raise ValueError(
                f"All audio arrays must have the same length. "
                f"Station {name} has length {len(audio)}, expected {data_len}"
            )

    # Create Data matrix
    data = np.zeros((len(receiver_positions), data_len))

    for i, station in enumerate(stations):
        # Subtract DC offset (detrend signal)
        audio = wav_dict[station]
        data[i, :] = np.round(audio - np.mean(audio))

    # track time to perform localization
    locstarttime = time.time()

    # Run MSRP
    power_map = compute_msrp(
        receiver_positions=receiver_positions,
        signals=data,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        time_delays=time_delays,
    )

    elapsed = time.time() - locstarttime
    print(f"Localized detection in {elapsed:.1f} seconds.")

    # Extract global maximum location
    max_idx = np.unravel_index(np.argmax(power_map), power_map.shape)
    location = np.array(
        [
            search_map["XAxis"][max_idx[0]],
            search_map["YAxis"][max_idx[1]],
            search_map["ZAxis"][max_idx[2]],
        ]
    )

    if plot:
        raise NotImplementedError("Plotting not yet implemented in Python version")
        # from matplotlib import pyplot as plt

        # x = search_map["XAxis"]
        # y = search_map["YAxis"]
        # # smap has shape (nx, ny, nz) with our new (x,y,z) indexing
        # # pcolormesh expects data[i,j] at position (x[j], y[i])
        # # so we need to transpose from (nx, ny) to (ny, nx) for the plot
        # plt.pcolormesh(x, y, power_map[:, :, 0].T, shading="auto")
        # plt.scatter(
        #     receiver_positions["Easting"],
        #     receiver_positions["Northing"],
        #     c="White",
        #     label="Receivers",
        # )

        # plt.scatter(
        #     location[0],
        #     location[1],
        #     c="Red",
        #     label="Estimated Position",
        # )
        # plt.legend()
        # plt.show()

    # Build output dictionary
    out = {"location": location, "max_power": np.max(power_map)}

    if keep_power_map:
        out["power_map"] = power_map

    return out
