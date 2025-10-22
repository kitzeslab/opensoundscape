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
from scipy.spatial import ConvexHull


class SpatialGrid:
    """
    Class for creating a grid of points for localizing sound events with methods that use a grid search approach.
    """

    def __init__(
        self,
        recorder_positions,
        sample_rate,
        resolution=1,
        margin=0,
        speed_of_sound=343,
        compute_time_intervals=True,
        pairwise_time_intervals=None,
        pair_idx_lookup=None,
    ):
        """
        Initialize a SpatialGrid object
        Args:
            recorder_positions: list of [x,y] or [x,y,z] positions of each recorder in meters
            sample_rate: sample rate of the audio in Hz.
            resolution: resolution of the grid in meters. Default is 1.
            margin: margin around the convex hull of the grid in meters. Will only attempt to localize events that are inside the grid + margin.
                    A negative margin will shrink the grid. Default is 0.
            speed_of_sound: speed of sound in meters per second. Default is 343 m/s.
            compute_time_intervals: whether to pre-compute time-delay intervals for each point in the grid. Default is True.
            time_intervals: precomputed time-delay intervals to use instead of computing them. Default is None.
            pairwise_time_intervals: precomputed pairwise time-delay intervals to use instead of computing them. Default is None.
        """
        self.recorder_positions = np.array(recorder_positions)
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.margin = margin
        self.dimensions = self.recorder_positions.shape[1]
        self.speed_of_sound = speed_of_sound
        self.grid = self._make_grid()

        if pairwise_time_intervals is not None:
            self.pairwise_time_intervals = pairwise_time_intervals
            self.pair_idx_lookup = pair_idx_lookup
        elif compute_time_intervals:
            self._make_time_intervals()
        else:
            self.pairwise_time_intervals = None
            self.pair_idx_lookup = None

    def get_pairwise_time_intervals(self, receiver_idx1, receiver_idx2):
        """
        Get the pairwise time-delay intervals for a specific pair of receivers
        Args:
            receiver_idx1: index of the first receiver
            receiver_idx2: index of the second receiver
        Returns:
            pairwise_time_intervals: the time-delay intervals for the specified pair of receivers
        """
        if self.pairwise_time_intervals is None:
            raise ValueError(
                "Time-delay intervals have not been computed for this SpatialGrid."
            )
        pair_idx = self.pair_idx_lookup.get((receiver_idx1, receiver_idx2))
        if pair_idx is None:
            raise ValueError("Invalid receiver indices.")
        return self.pairwise_time_intervals[:, pair_idx, :]

    def _make_grid(self, filter_to_convex_hull=True):
        """
        Create a grid of points for localizing sound events
        Returns:
            grid: a list of [x,y] or [x,y,z] positions of each point in the grid
        """

        # make a grid of all the points between the min and max possible coordinates of the recorder positions
        x = np.arange(
            np.floor(np.min(self.recorder_positions[:, 0])) - self.margin,
            np.ceil(np.max(self.recorder_positions[:, 0])) + self.margin,
            self.resolution,
        )
        y = np.arange(
            np.floor(np.min(self.recorder_positions[:, 1])) - self.margin,
            np.ceil(np.max(self.recorder_positions[:, 1])) + self.margin,
            self.resolution,
        )

        if self.dimensions == 2:
            grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        else:
            z = np.arange(
                np.floor(np.min(self.recorder_positions[:, 2])) - self.margin,
                np.ceil(np.max(self.recorder_positions[:, 2])) + self.margin,
                self.resolution,
            )
            grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        if filter_to_convex_hull:
            hull = ConvexHull(self.recorder_positions)
            # only keep the points that are inside the convex hull of the recorder positions
            # hull.equations is the equation of the hyperplane of each face of the convex hull
            # apply the equation of each face to the grid points to check if they are inside the convex hull
            eps = 1e-6
            # add a small epsilon to the margin to ensure that points on the edge of the convex hull are included
            mask = np.all(
                hull.equations[:, :-1].dot(grid.T) + hull.equations[:, -1][:, None]
                <= self.margin + eps,
                axis=0,
            )
            grid = grid[mask]

        return grid

    def subset(self, grid_points):
        """
        Create a copy of the SpatialGrid, filtering points and time_intervals to only those within the convex hull of the recorder positions
        Args:
            grid_points: array of shape (N, 2) or (N, 3) containing [x,y] or [x,y,z] positions of each grid point
        Returns:
            subset: SpatialGrid object for the reduced search grid
        """
        import copy

        subset_grid = copy.deepcopy(self)

        # find indices of self.recorder_positions, first match for each of grid_points
        grid_points = np.array(grid_points)[:, : self.recorder_positions.shape[1]]
        grid_point_idx_to_keep = [
            np.where((self.recorder_positions == point).all(axis=1))[0][0]
            for point in grid_points
        ]

        if self.dimensions == 2:
            hull = ConvexHull(grid_points[:, :2])
        else:
            hull = ConvexHull(grid_points)

        # only keep the points that are inside the convex hull of the recorder positions
        # hull.equations is the equation of the hyperplane of each face of the convex hull
        # apply the equation of each face to the grid points to check if they are inside the convex hull
        eps = 1e-6
        # add a small epsilon to the margin to ensure that points on the edge of the convex hull are included
        mask = np.all(
            hull.equations[:, :-1].dot(self.grid.T) + hull.equations[:, -1][:, None]
            <= self.margin + eps,
            axis=0,
        )
        subset_grid.grid = self.grid[mask]
        if subset_grid.pairwise_time_intervals is not None:
            # we need to subset the pairwise_time_intervals to only include the receiver pairs that are in the subset grid
            # TODO: check that this always creates the correct pair order for the recorder subset
            pair_mask = [
                idx1 in grid_point_idx_to_keep and idx2 in grid_point_idx_to_keep
                for idx1, idx2 in self.pair_idx_lookup.keys()
            ]
            subset_grid.pairwise_time_intervals = self.pairwise_time_intervals[
                :, mask, :
            ][:, :, pair_mask]

        return subset_grid

    def _make_time_intervals(self):
        """
        Pre-compute time-delay intervals for each point in the grid
        Returns:
            pairwise_time_intervals: a dict with keys 'T1' and 'T2' containing the valid time-delay intervals for each point in the grid
        """
        self.pairwise_time_intervals, receiver_idx_pairs = compute_time_delay_intervals(
            receiver_positions=self.recorder_positions,
            grid_positions=self.grid,
            sample_rate=self.sample_rate,
            speed_of_sound=self.speed_of_sound,
            resolution=self.resolution,
        )
        self.pair_idx_lookup = {
            (i, j): idx for idx, (i, j) in enumerate(receiver_idx_pairs.T)
        }


def compute_time_delay_intervals(
    receiver_positions, grid_positions, sample_rate, speed_of_sound, resolution
):
    """Pre-compute time-delays at search positions

    Args:
        receiver_positions: ndarray of shape (N, 3) containing [x, y, z] positions of N receivers
        (microphones) search_map: dict with keys 'resolution' and
            'XAxis', 'YAxis', 'ZAxis' (1D arrays of search grid coordinates)
        sample_rate: float, sample rate in Hz speed_of_sound: float, speed of sound in m/s

    Returns:
        dict containing InitData with precomputed valid time delay intervals in samples
        for each grid point and receiver pair
            The 'T1' and 'T2' arrays have shape (nx, ny, nz, NIJ) where:
            - nx, ny, nz are the spatial dimensions from search_map
            - NIJ is the number of receiver pairs

    Author: Sam Lapp, translated from Tim Huang (original Matlab implementation) and Richard Hedley
    (R implementation)
    """
    import time

    # start_time = time.time()

    receiver_positions = np.asarray(receiver_positions)
    n_receivers, dims = receiver_positions.shape

    sr = sample_rate  # for brevity

    # Number of pairwise comparisons of receivers
    n_receiver_pairs = n_receivers * (n_receivers - 1) // 2

    # Create list of all pairwise comparisons
    import itertools

    receiver_idx_pairs = np.array(
        list(itertools.combinations(range(n_receivers), 2)), dtype=int
    ).T
    T1 = np.zeros((len(grid_positions), n_receiver_pairs), dtype=int)
    T2 = np.zeros((len(grid_positions), n_receiver_pairs), dtype=int)

    # Iterate over all grid points using spatial indices
    for i, grid_position in enumerate(grid_positions):

        # Distance vector from location (x,y,z) to each of N receivers
        # Add small epsilon to avoid zero distances
        rel_receiver_position = grid_position - receiver_positions
        receiver_dist = np.sqrt(np.sum(rel_receiver_position**2, axis=1)) + 1e-6

        # Normalized directional differences divided by speed_of_sound (gradient of time delay)
        # eq 9
        gradient_tau = np.array(
            [
                (
                    (
                        grid_position[dim]
                        - receiver_positions[receiver_idx_pairs[1], dim]
                    )
                    / receiver_dist[receiver_idx_pairs[1]]
                    - (
                        grid_position[dim]
                        - receiver_positions[receiver_idx_pairs[0], dim]
                    )
                    / receiver_dist[receiver_idx_pairs[0]]
                )
                / speed_of_sound
                for dim in range(dims)
            ]
        )
        # grad_tau_x = (
        #     (grid_position[0] - receiver_positions[pairs[1], 0])
        #     / receiver_dist[pairs[1]]
        #     - (grid_position[0] - receiver_positions[pairs[0], 0])
        #     / receiver_dist[pairs[0]]
        # ) / speed_of_sound
        # grad_tau_y = (
        #     (grid_position[1] - receiver_positions[pairs[1], 1])
        #     / receiver_dist[pairs[1]]
        #     - (grid_position[1] - receiver_positions[pairs[0], 1])
        #     / receiver_dist[pairs[0]]
        # ) / speed_of_sound
        # grad_tau_z = (
        #     (grid_position[2] - receiver_positions[pairs[1], 2])
        #     / receiver_dist[pairs[1]]
        #     - (grid_position[2] - receiver_positions[pairs[0], 2])
        #     / receiver_dist[pairs[0]]
        # ) / speed_of_sound

        # magnitude of the gradient
        grad_mag = np.linalg.norm(gradient_tau, axis=0)

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
        # Calculate distance to boundary d of a cube with edge length resolution, following the gradient direction
        # eq 12
        # phi: azimuth angle, theta: elevation desceding from z-axis
        # x = r * sin(theta) * cos(phi), y = r * sin(theta) * sin(phi), z = r * cos(theta)
        # setting x,y,or z to resolution/2 gives
        # dist to box edge = min distance to intersect the x=resolution/2, y=resolution/2, z=resolution/2 planes
        # distance to plane in x: d_x = (resolution / 2) / np.abs(np.cos(phi) * np.sin(theta))
        # distance to plane in y: d_y = (resolution / 2) / np.abs(np.sin(phi) * np.sin(theta))
        # distance to plane in z: d_z = (resolution / 2) / np.abs(np.cos(theta))
        if dims < 3:
            theta = np.repeat(np.pi / 2, receiver_idx_pairs.shape[1])  # zero elevation
        else:
            theta = np.arccos(gradient_tau[2] / grad_mag)  # eq 13
        eps = 1e-12  # avoid divsion by zero; large vallues will be ignored because of min()
        phi = np.arctan2(gradient_tau[1], gradient_tau[0])  # eq 14
        normalized_directional_distances = np.vstack(
            [
                1 / abs(np.cos(phi) * np.sin(theta) + eps),
                1 / abs(np.sin(phi) * np.sin(theta) + eps),
                1 / abs(np.cos(theta) + eps),
            ]
        )
        # d: dist_to_finite_element_edge_along_gradient, for a cube with edge length resolution
        d = 0.5 * resolution * np.min(normalized_directional_distances, axis=0)

        # Compute time delays in seconds
        paired_distance_diff = (
            receiver_dist[receiver_idx_pairs[1]] - receiver_dist[receiver_idx_pairs[0]]
        )
        dT1 = (paired_distance_diff) / speed_of_sound - grad_mag * d
        dT2 = (paired_distance_diff) / speed_of_sound + grad_mag * d

        # convert time to samples
        dT1 = np.round(dT1 * sr)
        dT2 = np.round(dT2 * sr)

        # Store in 3D arrays
        T1[i, :] = dT1.astype(int)  # .flatten()
        T2[i, :] = dT2.astype(int)  # .flatten()

        # elapsed = time.time() - start_time
        # print(f"Computed pairwise arrival times in {elapsed:.1f} seconds")

        pairwise_time_delay_intervals = np.array([T1, T2])

    return pairwise_time_delay_intervals, receiver_idx_pairs


def compute_msrp(
    receiver_positions,
    signals,
    sample_rate,
    freq_low,
    freq_high,
    time_delay_intervals,
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
            time delay indices for each search location for each receiver pair
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

    # Add signal_len-1 to get 0-based center index corresponding to gcc output
    td = time_delay_intervals.copy() + (signals.shape[1] - 1)

    # td["T1"] = td["T1"] + (signals.shape[1] - 1)
    # td["T2"] = td["T2"] + (signals.shape[1] - 1)

    # clamp time_delays to valid range [0, cc_len-1]
    # td["T1"] = np.clip(td["T1"], 0, cc_len - 1)
    # td["T2"] = np.clip(td["T2"], 0, cc_len - 1)
    td = np.clip(td, 0, cc_len - 1)

    k = 0
    pairwise_cc = np.zeros((n_receiver_pairs, cc_len))  # Preallocate cross-correlation

    # Compute generalized cross-correlation for all pairs
    for grid_idx in range(n_receivers - 1):
        for j in range(grid_idx + 1, n_receivers):
            # Use the gcc function from signal_processing.py
            # with frequency range filtering; note ordering convention
            gcc_result = gcc(
                signals[j, :],  # signal
                signals[grid_idx, :],  # reference signal
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
    grid_size = len(td[0])  # Total number of grid points

    # Initialize steered response power array with spatial dimensions
    srp = np.zeros(grid_size)

    # Calculate steered response power for each spatial location
    # by integrating over valid time delay ranges
    for grid_idx in range(grid_size):
        sub1 = []
        sub2 = []
        for pair_idx in range(n_receiver_pairs):
            # Build indices for valid time delay range
            # They should already be clamped to [0, ML2-1]
            t1 = td[0][grid_idx, pair_idx]
            t2 = td[1][grid_idx, pair_idx]
            if t2 >= t1:  # >0 interval to sum over
                n_indices = t2 - t1 + 1
                sub1.extend([pair_idx] * n_indices)
                sub2.extend(range(t1, t2 + 1))

        # Extract relevant correlation values and sum
        # extensions of the approach use different aggregation
        if len(sub1) > 0:
            tR = pairwise_cc[sub1, sub2]
            srp[grid_idx] = np.sum(tR)
        else:
            srp[grid_idx] = 0

    return srp


def localize(
    signals,
    receiver_positions,
    search_map,
    freq_low=2000,
    freq_high=8000,
    keep_power_map=True,
):
    """Localize a sound source using MSRP algorithm.

    Main function to localize a sound source given synchronized audio recordings from multiple
    stations using the MSRP (Modified Steered Response Power) algorithm.

    Args:
        signals: np.array of shape (N, signal lenth) containing audio samples from each receiver
        receiver_positions: np.array of shape (N, 3) containing [x, y, z] positions of N receivers
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

    # detrend, subtracting dc offset
    signals = np.asarray(signals) - np.mean(signals, axis=1, keepdims=True)

    # track time to perform localization
    locstarttime = time.time()

    if search_map.pairwise_time_intervals is None:
        search_map._make_time_intervals()

    # Run MSRP
    power_map = compute_msrp(
        receiver_positions=receiver_positions,
        signals=signals,
        sample_rate=search_map.sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        time_delay_intervals=search_map.pairwise_time_intervals,
    )

    elapsed = time.time() - locstarttime
    # print(f"Localized detection in {elapsed:.1f} seconds.")

    # Extract global maximum location
    location = search_map.grid[np.argmax(power_map)]

    # Build output dictionary
    out = {"location": location, "max_power": np.max(power_map)}

    if keep_power_map:
        out["power_map"] = power_map
    return out
