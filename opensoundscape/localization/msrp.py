"""MSRP (Maximum Source Range Power) localization algorithm

This module contains Python translations of the MSRP localization functions
originally implemented in Matlab by Tim Huang, then in R by Richard Hedley (locaR package)

#TODO: refactor to use list of (x,y,z) positions instead of requiring cubic grid
"""

import itertools
import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal
from opensoundscape.signal_processing import gcc
from scipy.spatial import ConvexHull


class SearchMap:
    """
    Class for creating a grid of points for localizing sound events with methods that use a grid search approach.
    """

    def __init__(
        self,
        receiver_positions,  # refactor to df {receiver_id|position}
        sample_rate,
        resolution,
        margin=0,
        speed_of_sound=343,
        compute_time_intervals=True,
    ):
        """
        Initialize a SearchMap object with a cubic grid of points for localizing sound events.
        Args:
            recorder_positions: DataFrame with columns [x,y,(z)] for each receiver's position in meters
            sample_rate: sample rate of the audio in Hz.
            resolution: resolution of the grid in meters. Default is 1.
            margin: margin around the convex hull of the grid in meters. Will only attempt to localize events that are inside the grid + margin.
                    A negative margin will shrink the grid. Default is 0.
            speed_of_sound: speed of sound in meters per second. Default is 343 m/s.
            compute_time_intervals: whether to pre-compute time-delay intervals for each point in the grid. Default is True.
        """
        self.receiver_positions = receiver_positions
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.margin = margin
        self.dimensions = self.receiver_positions.shape[1]  # 2d or 3d spatial grid
        self.speed_of_sound = speed_of_sound
        self._make_grid()  # creates self.search_points list of coordinates

        if compute_time_intervals:
            self._make_time_intervals()
        else:
            self.time_delay_min = None
            self.time_delay_max = None

    def _make_grid(self, filter_to_convex_hull=True):
        """
        Create a grid of points for localizing sound events
        Returns:
            grid: a list of [x,y] or [x,y,z] positions of each point in the grid
        """

        # make a grid of all the points between the min and max possible coordinates of the recorder positions
        x = np.arange(
            np.floor(np.min(self.receiver_positions.iloc[:, 0])) - self.margin,
            np.ceil(np.max(self.receiver_positions.iloc[:, 0])) + self.margin,
            self.resolution,
        )
        y = np.arange(
            np.floor(np.min(self.receiver_positions.iloc[:, 1])) - self.margin,
            np.ceil(np.max(self.receiver_positions.iloc[:, 1])) + self.margin,
            self.resolution,
        )

        if self.dimensions == 2:
            search_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        else:
            z = np.arange(
                np.floor(np.min(self.receiver_positions.iloc[:, 2])) - self.margin,
                np.ceil(np.max(self.receiver_positions.iloc[:, 2])) + self.margin,
                self.resolution,
            )

            search_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        if filter_to_convex_hull:
            search_points = filter_points_to_convex_hull(
                search_points, self.receiver_positions, self.margin
            )

        self.search_points = pd.DataFrame(
            search_points, columns=self.receiver_positions.columns
        )

    def subset(self, receivers, margin=0):
        """
        Create a copy of the SpatialGrid, filtering search grid to the convex hull of a subset of receivers.

        Args:
            receivers: list of receiver ids to retain
        Returns:
            subset: SpatialGrid object for the reduced search grid
        """
        import copy

        subset_grid = copy.deepcopy(self)

        # what do we do when there are pairs in the opposite order?

        # find indices of receiver pairs in the original grid
        # and create list of indices for [pairs of receiver indices]
        rec_pairs = list(itertools.combinations(receivers, 2))

        receiver_positions = self.receiver_positions.loc[receivers].values

        if self.dimensions == 2:
            hull = ConvexHull(receiver_positions[:, :2])
        else:
            hull = ConvexHull(receiver_positions)

        # only keep the points that are inside the convex hull of the recorder positions
        # hull.equations is the equation of the hyperplane of each face of the convex hull
        # apply the equation of each face to the grid points to check if they are inside the convex hull
        eps = 1e-6
        # add a small epsilon to the margin to ensure that points on the edge of the convex hull are included
        point_mask = np.all(
            hull.equations[:, :-1].dot(self.search_points.values.T)
            + hull.equations[:, -1][:, None]
            <= margin + eps,
            axis=0,
        )
        subset_grid.search_points = self.search_points[point_mask]

        if subset_grid.time_delay_max is not None:
            # subset pairwise time intervals by included points (index) then by receiver pairs (columns)
            subset_grid.time_delay_max = self.time_delay_max[point_mask][rec_pairs]
        if subset_grid.time_delay_min is not None:
            # subset pairwise time intervals by included points (index) then by receiver pairs (columns)
            subset_grid.time_delay_min = self.time_delay_min[point_mask][rec_pairs]

        return subset_grid

    def _make_time_intervals(self):  # TODO refactor for explicit receiver lookup
        """
        Pre-compute time-delay intervals for each point in the grid
        """
        self.time_delay_min, self.time_delay_max = compute_time_delay_intervals(
            receiver_positions=self.receiver_positions,
            grid_positions=self.search_points,
            sample_rate=self.sample_rate,
            speed_of_sound=self.speed_of_sound,
            resolution=self.resolution,
        )


def filter_points_to_convex_hull(points, boundary_points, margin=0):
    hull = ConvexHull(boundary_points)
    # only keep the points that are inside the convex hull of the boundary points + margin
    # hull.equations is the equation of the hyperplane of each face of the convex hull
    # apply the equation of each face to the grid points to check if they are inside the convex hull
    eps = 1e-6
    # add a small epsilon to the margin to ensure that points on the edge of the convex hull are included
    mask = np.all(
        hull.equations[:, :-1].dot(points.T) + hull.equations[:, -1][:, None]
        <= margin + eps,
        axis=0,
    )
    return points[mask]


def compute_time_delay_intervals(
    receiver_positions, grid_positions, sample_rate, speed_of_sound, resolution
):
    """Pre-compute time-delay boundaries for pairs of receivers at cubic search grid cells

    Args:
        receiver_positions: pd.DataFrame with receiver positions in meters, shape (N, 2) or (N, 3)
        grid_positions: pd.DataFrame with grid positions in meters, shape (M, 2) or (M, 3)
        sample_rate: float, sample rate in Hz
        speed_of_sound: float, speed of sound in m/s

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

    rec_coords = receiver_positions.values
    n_receivers, dims = rec_coords.shape

    sr = sample_rate  # for brevity

    # Create list of all pairwise comparisons
    # NOTE: was .combinations to exploit symmetry, but using .permutations so that we can look up (a,b) and (b,a)
    receiver_pairs = np.array(
        list(itertools.permutations(receiver_positions.index, 2))
    ).T
    # Number of pairwise comparisons of receivers
    n_receiver_pairs = receiver_pairs.shape[1]

    T1 = np.zeros((len(grid_positions), n_receiver_pairs), dtype=int)
    T2 = np.zeros((len(grid_positions), n_receiver_pairs), dtype=int)

    # Iterate over all grid points to compute time delay intervals for each receiver pair
    for grid_idx in grid_positions.index:
        grid_position = grid_positions.loc[grid_idx].values
        # Distance vector from location (x,y,z) to each of N receivers
        # Add small epsilon to avoid zero distances
        rel_receiver_positions = grid_position - rec_coords
        receiver_dist = pd.Series(
            np.sqrt(np.sum(rel_receiver_positions**2, axis=1)) + 1e-6,
            index=receiver_positions.index,
        )

        # Normalized directional differences divided by speed_of_sound (gradient of time delay)
        # eq 9
        # this is vectorized over all receiver pairs
        gradient_tau = np.array(
            [
                (
                    (
                        (
                            grid_position[dim]
                            - receiver_positions.loc[receiver_pairs[1]].iloc[:, dim]
                        )
                        / receiver_dist.loc[receiver_pairs[1]]
                    ).values
                    - (
                        (
                            grid_position[dim]
                            - receiver_positions.loc[receiver_pairs[0]].iloc[:, dim]
                        )
                        / receiver_dist.loc[receiver_pairs[0]]
                    ).values
                )
                / speed_of_sound
                for dim in range(dims)
            ]
        )

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
            theta = np.repeat(np.pi / 2, receiver_pairs.shape[1])  # zero elevation
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
            receiver_dist.loc[receiver_pairs[1]].values
            - receiver_dist.loc[receiver_pairs[0]].values
        )
        dT1 = (paired_distance_diff) / speed_of_sound - grad_mag * d
        dT2 = (paired_distance_diff) / speed_of_sound + grad_mag * d

        # convert time to samples
        dT1 = np.round(dT1 * sr)
        dT2 = np.round(dT2 * sr)

        # Store in 3D arrays
        T1[grid_idx, :] = dT1.astype(int)  # .flatten()
        T2[grid_idx, :] = dT2.astype(int)  # .flatten()

        # elapsed = time.time() - start_time
        # print(f"Computed pairwise arrival times in {elapsed:.1f} seconds")

        # pairwise_time_delay_intervals = np.array([T1, T2])

    # make 2 dfs with index (grid coord tuples), columns (receiver pair tuples)
    t1_df = pd.DataFrame(
        T1,
        columns=[(i, j) for i, j in receiver_pairs.T],
        index=[tuple(pos) for pos in grid_positions.values],
    )
    t2_df = pd.DataFrame(
        T2,
        columns=[(i, j) for i, j in receiver_pairs.T],
        index=[tuple(pos) for pos in grid_positions.values],
    )
    return t1_df, t2_df


def compute_msrp(
    signals,
    search_map,
    freq_low,
    freq_high,
    cc_filter="phat",
    aggregation_fn=np.sum,
):
    """Calculate likelihood of sound sources at each search location.

    This function uses the InitData and other info to calculate the likelihood
    of sound sources coming from each location using generalized cross-correlation
    with the Hilbert transform.

    Uses "level 2" implementation of original Matlab implementation

    Args:
        receiver_positions: pd.DataFrame with receiver positions in meters, shape (N, 2) or (N, 3)
            index contains receiver ID
        signals: dictionary of receiver : np.array of shape (signal length,) containing audio samples
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
    signal_len = len(next(iter(signals.values())))
    cc_len = signal_len * 2 - 1  # Length of cross-correlation result

    # Add signal_len-1 to get 0-based center index corresponding to gcc output
    t0 = search_map.time_delay_min.copy() + (signal_len - 1)
    t0 = np.clip(t0, 0, cc_len - 1)
    t1 = search_map.time_delay_max.copy() + (signal_len - 1)
    t1 = np.clip(t1, 0, cc_len - 1)

    # Initialize array to hold pairwise cross-correlations for each receiver pair
    # that will be used in this localization
    receiver_pairs = list(itertools.combinations(list(signals.keys()), 2))
    pairwise_cc = {}

    # chek all pairs of receivers are included in time delay data
    for rec_pair in receiver_pairs:
        if rec_pair not in t0.columns:
            raise ValueError(f"Missing time delay data for receiver pair {rec_pair}")

    # Compute generalized cross-correlation for all pairs
    for receiver_pair in receiver_pairs:
        # Use the gcc function from signal_processing.py
        # with frequency range filtering; note ordering convention
        gcc_result = gcc(
            signals[receiver_pair[1]],  # target signal
            signals[receiver_pair[0]],  # reference signal
            cc_filter=cc_filter,  # phat is used in original implementation
            frequency_range=(freq_low, freq_high),
            sample_rate=search_map.sample_rate,
        )

        # Apply Hilbert transform and take absolute value
        # Note: scipy.signal.hilbert returns the analytic signal
        analytic_signal = scipy_signal.hilbert(gcc_result)
        pairwise_cc[receiver_pair] = np.abs(analytic_signal)

    # Initialize steered response power array with spatial dimensions
    srp = pd.Series(0, index=t0.index)  # grid points as index

    # Calculate steered response power for each spatial location
    # by integrating over valid time delay ranges
    for grid_point in t0.index:
        cell_ccs = []
        for rec_pair in receiver_pairs:
            # Build indices for valid time delay range
            # They should already be clamped to [0, ML2-1]
            tmin = t0.at[grid_point, rec_pair]
            tmax = t1.at[grid_point, rec_pair]
            if tmax >= tmin:
                # >0 interval of valid time delays for this receiver pair in this grid cell
                cell_ccs.extend(pairwise_cc[rec_pair][tmin : tmax + 1])

        # Aggregation: default is to sum the ccs for this cell over all receiver pairs
        srp.at[grid_point] = aggregation_fn(cell_ccs)

    return srp


def localize(
    signals,
    search_map,
    freq_low=None,
    freq_high=None,
    keep_maps=True,
    cc_filter="phat",
    aggregation_fn=np.sum,
    convex_hull_margin=0,
    detrend=True,
):
    """Localize a sound source using MSRP algorithm.

    Main function to localize a sound source given synchronized audio recordings from multiple
    stations using the MSRP (Modified Steered Response Power) algorithm.

    Args:
        signals: dict of receiver_id : np.array of shape (signal length,) containing audio samples
        search_map: SearchMap object containing grid points and precomputed time delay intervals
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
        convex_hull_margin: float, margin in meters around convex hull of receivers to search
            - pass None to disable filtering to convex hull

    Returns:
        dict containing:
            - 'location': DataFrame with Easting, Northing, Elevation, Power
            - 'smap': (if keep_search_map=True) 3D likelihood array

    Author: Sam Lapp (Python implementation); Richard Hedley (R implementation); Tim Huang (original
    Matlab implementation);
    """
    import time

    # detrend, subtracting dc offset # TODO do we want/need this?
    if detrend:
        signals = {
            receiver_id: np.asarray(signal) - np.mean(signal)
            for receiver_id, signal in signals.items()
        }

    # track time to perform localization
    # locstarttime = time.time()

    if convex_hull_margin is not None:
        # filter search map to convex hull of receiver positions + margin
        search_map = search_map.subset(
            receivers=signals.keys(),
            margin=convex_hull_margin,
        )

    if search_map.time_delay_min is None or search_map.time_delay_max is None:
        search_map._make_time_intervals()

    # Run MSRP
    power_map = compute_msrp(
        signals=signals,
        search_map=search_map,
        freq_low=freq_low,
        freq_high=freq_high,
        cc_filter=cc_filter,
        aggregation_fn=aggregation_fn,
    )

    # elapsed = time.time() - locstarttime
    # print(f"Localized detection in {elapsed:.1f} seconds.")

    # Extract global maximum location
    location = power_map.idxmax()

    # Build output dictionary
    out = {"location": location, "max_power": np.max(power_map)}

    if keep_maps:
        out["power_map"] = power_map
        out["search_map"] = search_map
    return out
