"""Tools for localizing audio events from synchronized recording arrays"""
import warnings
import numpy as np
import pandas as pd
from opensoundscape.audio import Audio
from scipy.signal import correlate, correlation_lags


def gcc(x, y, max_delay_samples=None, filter="phat", epsilon=0.01):
    """
    GCC implementation based on Knapp and Carter - code adapted from
    github.com/axeber01/ngcc
    Args:
        x: 1d numpy array of audio samples
        y: 1d numpy array of audio samples
        max_delay_samples: maximum possible delay between the 2 signals in max_delay_samples
        filter: which filter to use in the gcc.
            'phat' - Phase transform,
            'roth',
            'scot' - Smoothed Coherence Transform,
            'ht' - Hannan and Thomson
        epsilon = used to ensure denominator is non-zero.
    """
    n = x.shape[0] + y.shape[0]

    # Generalized Cross Correlation Phase Transform
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    Gxy = X * np.conj(Y)

    if filter == "phat":
        phi = 1 / (np.abs(Gxy) + epsilon)

    elif filter == "roth":
        phi = 1 / (X * torch.conj(X) + epsilon)

    elif filter == "scot":
        Gxx = X * np.conj(X)
        Gyy = Y * np.conj(Y)
        phi = 1 / (np.sqrt(X * Y) + epsilon)

    elif filter == "ht":
        Gxx = X * np.conj(X)
        Gyy = Y * np.conj(Y)
        gamma = Gxy / np.sqrt(Gxx * Gxy)
        phi = np.abs(gamma) ** 2 / (np.abs(Gxy) * (1 - gamma) ** 2 + epsilon)
    elif filter == "cc":
        phi = 1.0
    else:
        raise ValueError(
            "Unsupported filter. Must be one of: 'ht', 'phat', 'roth','scot'"
        )

    # set the max delay in number of samples
    if max_delay_samples:
        max_delay_samples = np.minimum(max_delay_samples, int(n / 2))
    else:
        max_delay_samples = int(n / 2)

    cc = np.fft.irfft(Gxy * phi, n)

    return cc


# make a class that we will use to contain a model object, list of files and thresholds
# this will be called Localizer
# we will use this class for localizing sound sources from synchronized audio files
class Localizer:
    def __init__(
        self,
        model,
        files,
        aru_coords,
        sample_rate,
        min_number_of_receivers,
        max_distance_between_receivers,
        thresholds=None,
        predictions=None,
        bandpass_ranges=None,
        max_delay=None,
        cc_threshold=0,
    ):
        # initialize the class
        # model is a trained opensoundscape model
        # files is a list of synchronized audio files
        # aru_coords is a dictionary of aru coordinates, with key aru file path, and value (x,y) coordinates
        # thresholds is a dictionary of thresholds for each class
        # predictions is a pandas dataframe of predictions
        self.model = model
        self.files = files
        self.aru_coords = aru_coords
        self.thresholds = thresholds
        self.SAMPLE_RATE = sample_rate
        self.min_number_of_receivers = min_number_of_receivers
        self.max_distance_between_receivers = max_distance_between_receivers
        self.predictions = predictions
        self.bandpass_ranges = bandpass_ranges
        self.max_delay = max_delay
        self.cc_threshold = cc_threshold

        # initialize the below intermediates as None. #TODO: work out how to do this correctly
        self.detections = None
        self.cross_correlations = None
        self.filtered_cross_correlations = None

    def get_predictions(self):
        # get CNN predictions from synchronized audio files
        # return a pandas dataframe with the results
        if self.predictions is None:
            self.predictions = self.model.predict(self.files, activation_layer=None)
        else:
            raise UserWarning(
                "Predictions already exist - set predictions to None if you want to re-run predictions"
            )
        return self.predictions

    def threshold_predictions(self):
        # use a set of thresholds to filter the predictions
        if self.predictions is None:
            print("No predictions exist - running predictions")
            self.get_predictions()
        all_sp_detections = []
        for species in self.predictions.columns:
            df = self.predictions.loc[:, [species]]  # must be a dataframe
            detections = Localizer._get_detections(
                df, cnn_score_threshold=self.thresholds[species]
            )
            grouped_detections = Localizer._group_detections(
                detections,
                self.aru_coords,
                self.min_number_of_receivers,
                self.max_distance_between_receivers,
            )
            grouped_detections["species"] = species
            all_sp_detections.append(grouped_detections)
        detections_df = pd.concat(all_sp_detections)
        self.detections = detections_df
        return detections_df

    def cross_correlate(self):
        # cross correlate the predictions
        # return a pandas dataframe with the results
        if self.bandpass_ranges is None:
            raise UserWarning("No bandpass range specified")
        if self.max_delay is None:
            raise UserWarning("No max delay specified")
        if self.detections is None:
            print("No detections exist - running threshold_predictions")
            self.threshold_predictions()
        # get the cross-correlations
        all_ccs = []
        all_tds = []
        for index, row in self.detections.iterrows():
            species = row["species"]
            cc, td = Localizer._get_cross_correlations(
                reference_file=row["reference_file"],
                other_files=row["other_files"],
                start_time=row["time"][0],
                end_time=row["time"][1],
                bandpass_range=self.bandpass_ranges[species],
                max_delay=self.max_delay,
                SAMPLE_RATE=44100,
            )
            all_ccs.append(cc)
            all_tds.append(td)
        self.cross_correlations = self.detections.copy()
        self.cross_correlations["cross_correlations"] = all_ccs
        self.cross_correlations["time_delays"] = all_tds
        return self.cross_correlations

    def filter_cross_correlations(self):
        # filter the cross correlations
        # return a pandas dataframe with the results
        if self.cross_correlations is None:
            print("No cross correlations exist - running cross_correlate")
            self.cross_correlate()
        # filter the cross-correlations
        above_threshold = [
            cc > self.cc_threshold
            for cc in self.cross_correlations["cross_correlations"]
        ]

        n_before = len(self.cross_correlations)  # number of rows before filtering

        filtered_ccs = []
        filtered_files = []
        filtered_tdoas = []
        for i in range(len(self.cross_correlations)):
            mask = above_threshold[i]
            cc = self.cross_correlations["cross_correlations"].iloc[i]
            other_files = np.array(self.cross_correlations["other_files"].iloc[i])
            tdoa = np.array(self.cross_correlations["time_delays"].iloc[i])

            filtered_ccs.append(cc[mask])
            filtered_files.append(other_files[mask])
            filtered_tdoas.append(tdoa[mask])

        filtered_cross_correlations = self.cross_correlations.copy()

        filtered_cross_correlations["cross_correlations"] = filtered_ccs
        filtered_cross_correlations["other_files"] = filtered_files
        filtered_cross_correlations["time_delays"] = filtered_tdoas

        # Filter by the cc scores. If less than min_number_of_receivers have cc_score above threshold, drop them.
        ccs = [
            np.array(scores)
            for scores in filtered_cross_correlations["cross_correlations"]
        ]
        num_ccs_above_threshold = [sum(a > self.cc_threshold) for a in ccs]
        mask = np.array(num_ccs_above_threshold) >= self.min_number_of_receivers - 1
        filtered_cross_correlations = filtered_cross_correlations[mask]

        n_after = len(filtered_cross_correlations)  # number of rows after filtering
        print(f"{n_before - n_after} rows deleted")
        self.filtered_cross_correlations = filtered_cross_correlations
        return filtered_cross_correlations

    def localize(self, algorithm="gillette"):
        # localize the detections
        # return a pandas dataframe with the results
        # TODO: make work for 3d

        localized = self.filtered_cross_correlations.copy()
        locations = []
        if self.filtered_cross_correlations is None:
            print(
                "No filtered cross_correlations exist - running filter_cross_correlations"
            )
            self.filter_cross_correlations()
        if algorithm == "gillette":
            # localize using gillette

            for index, row in self.filtered_cross_correlations.iterrows():
                reference = row["reference_file"]
                others = row["other_files"]
                reference_coords = self.aru_coords.loc[reference]
                others_coords = [self.aru_coords.loc[i] for i in others]
                all_coords = [reference_coords] + others_coords
                # add 0 tdoa for reference receiver
                delays = np.insert(row["time_delays"], 0, 0)

                location, _, _ = gillette_localize(all_coords, delays)
                locations.append(location)
            localized["predicted_x"] = [locations[i][0] for i in range(len(locations))]
            localized["predicted_y"] = [locations[i][1] for i in range(len(locations))]
            localized["gillette_error"] = ["Error" for i in range(len(locations))]
        elif algorithm == "soundfinder":

            for index, row in self.filtered_cross_correlations.iterrows():
                reference = row["reference_file"]
                others = row["other_files"]
                reference_coords = self.aru_coords.loc[reference]
                others_coords = [self.aru_coords.loc[i] for i in others]
                all_coords = [reference_coords] + others_coords
                # add 0 tdoa for reference receiver
                delays = np.insert(row["time_delays"], 0, 0)

                location = soundfinder(all_coords, delays)
                locations.append(location)
            localized["predicted_x"] = [locations[i][0] for i in range(len(locations))]
            localized["predicted_y"] = [locations[i][1] for i in range(len(locations))]
            localized["pseudorange_error"] = [
                locations[i][2] for i in range(len(locations))
            ]
        else:
            raise UserWarning("Algorithm not recognized")
        self.locations = localized
        return localized

    def _get_cross_correlations(
        reference_file,
        other_files,
        start_time,
        end_time,
        bandpass_range,
        max_delay,
        SAMPLE_RATE,
    ):
        """
        Gets the maximal cross correlations and the time-delay (in s) corresponding to that cross correlation between
        the reference_file and other_files. Setting max_delay ensures that only cross-correlations
        +/- a certain time-delay are returned. i.e if a sound can be a maximum of +/-
        ----
        args:
            reference_file: Path to reference file.
            other_files: List of paths to the other files which will be cross-correlated against reference_file
            start_time: start of time segment (in seconds) to be cross-correlated
            end_time: end of time segment (in seconds) to be cross-correlated.
            bandpass_range: [lower, higher] of bandpass range.
            max_delay: the maximum time (in seconds) to return cross_correlations for. i.e. if the best cross correlation
                        occurs for a time-delay greater than max_delay, the function will not return it, instead it will return
                        the maximal cross correlation within +/- max_delay
            SAMPLE_RATE: the sampling rate of the audio.
        returns:
            ccs: list of maximal cross-correlations for each pair of files.
            time_differences: list of time differences (in seconds) that yield the maximal cross-correlation.
        """
        lower = min(bandpass_range)
        higher = max(bandpass_range)

        reference_audio = Audio.from_file(
            reference_file, offset=start_time, duration=end_time - start_time
        ).bandpass(lower, higher, order=9)
        other_audio = [
            Audio.from_file(
                i, offset=start_time, duration=end_time - start_time
            ).bandpass(lower, higher, order=9)
            for i in other_files
        ]

        max_lag = int(
            max_delay * SAMPLE_RATE
        )  # Convert max_delay (in s) to max_lag in samples

        ccs = np.zeros(len(other_audio))
        time_difference = np.zeros(len(other_audio))
        for index, audio_object in enumerate(other_audio):
            ff = reference_audio.samples
            sf = audio_object.samples

            # TODO: Normalize these, so cross-correlation will return values -1<cc<1
            # TODO: verify this makes sense, could there be some floating point issues with this? Is it the right kind
            # of normalization
            ff = ff / np.std(ff)
            sf = sf / np.std(sf)

            cc = correlate(ff, sf, mode="same")  # correlations are per sample
            cc /= min(len(ff), len(sf))
            lags = correlation_lags(ff.size, sf.size, mode="same")

            # slice cc and lags, so we only look at cross_correlations that are between -max_lag and +max_lag
            lower_limit = int(len(cc) / 2 - max_lag)
            upper_limit = int(len(cc) / 2 + max_lag)

            cc = cc[lower_limit:upper_limit]
            lags = lags[lower_limit:upper_limit]

            # from IPython.core.debugger import Pdb; Pdb().set_trace()
            max_cc = np.max(cc)
            lag = -lags[
                np.argmax(cc)
            ]  # in ties (>2 ccs with same max value), argmax returns the first.
            time_difference[index] = lag
            ccs[index] = max_cc
        time_difference = [i / SAMPLE_RATE for i in time_difference]

        return ccs, time_difference

    def _get_detections(predictions_df, cnn_score_threshold):
        """
        Takes the predictions_df of CNN scores *FOR A SINGLE SPECIES*, chooses only detections > cnn_score_threshold
        and outputs a dictionary of times at which events were detected, and the ARU files they were detected in.
        args:
            predictions_array: a dataframe with multi-index of (file, start, end) with a column that is values for model predictions
            *FOR A SINGLE SPECIES*
            cnn_score_threshold: the minimum CNN score needed for a time-window to be considered a detection.
        returns:
            A dictionary of predictions, with key (start_time, end_time), and value list of files with detection triggered
            e.g. {(0.0,2.0): [ARU_0.mp3. ARU_1.mp3]}
        """
        # get the detections from the predictions
        # Threshold the scores to above cnn_score_threshold
        booleans = (
            predictions_df.loc[:, :, :] > cnn_score_threshold
        )  # find rows above threshold
        indices = (
            booleans[booleans].dropna().index
        )  # choose just those rows. dropna required to drop the others
        recorders = indices.get_level_values(
            0
        )  # get the list of recorders out of the multi-index
        indices = indices.droplevel(level=0)  # drop the recorders

        dataframe = pd.DataFrame(
            data=recorders, index=indices
        )  # df with index (start_time, end_time)
        dataframe = (
            dataframe.sort_index()
        )  # done to ensure speed-up and not get performancewarning
        recorders_list = []
        for idx in dataframe.index.unique():
            recorders_in_time = dataframe.loc[idx].values
            recorders_in_time = [
                i[0] for i in recorders_in_time
            ]  # to get recorder path string out of numpy array
            recorders_list.append(recorders_in_time)
        return dict(zip(dataframe.index.unique(), recorders_list))

    def _group_detections(
        detections, aru_coords, min_number_of_receivers, max_distance_between_receivers
    ):
        """
        Takes the detections dictionary and groups detections that are within max_distance_between_receivers of each other.
        args:
            detections: a dictionary of detections, with key (start_time, end_time), and value list of files with detection triggered
            aru_coords: a dictionary of aru coordinates, with key aru file path, and value (x,y) coordinates
            max_distance_between_receivers: the maximum distance between recorders to consider a detection as a single event
        returns:
            A dictionary of grouped detections, with key (start_time, end_time), and value list of files with detection triggered
            e.g. {(0.0,2.0): [ARU_0.mp3. ARU_1.mp3]}
        """
        # group detections that are within max_distance_between_receivers of each other
        # return a dictionary of grouped detections
        # get the coordinates of the recorders
        # get the distance between recorders
        # if the distance is less than max_distance_between_receivers, group the detections
        from itertools import product

        # Group recorders based on being within < max_distance_between_receivers.
        # recorders_in_distance is dictionary in
        # form {ARU_0.mp3: [ARU_1.mp3, ARU_2.mp3...] for all recorders within max_distance_between_receivers }
        recorders_in_distance = dict()

        aru_files = aru_coords.index
        for aru in aru_files:  # loop over the aru files
            pos_aru = np.array(aru_coords.loc[aru])
            other_arus = np.array(aru_coords)
            distances = other_arus - pos_aru
            euclid_distances = [np.linalg.norm(d) for d in distances]

            mask = [
                0 <= i <= max_distance_between_receivers for i in euclid_distances
            ]  # boolean mask
            recorders_in_distance[aru] = list(aru_files[mask])

        times = []
        reference_files = []
        other_files = []

        for time_segment in detections.keys():  # iterate through all the time-segments
            for file in detections[
                time_segment
            ]:  # iterate through each file with a call detected in this time-segment
                reference = file  # set this file to be reference
                others = [
                    f for f in detections[time_segment] if f != reference
                ]  # All the other ARUs
                others_in_distance = [
                    aru for aru in others if aru in recorders_in_distance[reference]
                ]  # ARUs close enough

                if (
                    len(others_in_distance) + 1 >= min_number_of_receivers
                ):  # minimum number of ARUs needed to localize.
                    times.append(time_segment)
                    reference_files.append(reference)
                    other_files.append(others_in_distance)

        grouped_detections = pd.DataFrame(
            data=zip(times, reference_files, other_files),
            columns=["time", "reference_file", "other_files"],
        )
        return grouped_detections


def calc_speed_of_sound(temperature=20):
    """
    Calculate speed of sound in meters per second

    Calculate speed of sound for a given temperature
    in Celsius (Humidity has a negligible
    effect on speed of sound and so this functionality
    is not implemented)

    Args:
        temperature: ambient temperature in Celsius

    Returns:
        the speed of sound in meters per second
    """
    return 331.3 * np.sqrt(1 + float(temperature) / 273.15)


def lorentz_ip(u, v=None):
    """
    Compute Lorentz inner product of two vectors

    For vectors `u` and `v`, the
    Lorentz inner product for 3-dimensional case is defined as

        u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3]

    Or, for 2-dimensional case as

        u[0]*v[0] + u[1]*v[1] - u[2]*v[2]

    Args:
        u: vector with shape either (3,) or (4,)
        v: vector with same shape as x1; if None (default), sets v = u

    Returns:
        float: value of Lorentz IP"""
    if v is None:
        v = u

    if len(u) == 3 and len(v) == 3:
        c = [1, 1, -1]
        return sum([u[i] * v[i] * c[i] for i in range(len(u))])
    elif len(u) == 4 and len(v) == 4:
        c = [1, 1, 1, -1]
        return sum([u[i] * v[i] * c[i] for i in range(len(u))])

    return ValueError(f"length of x should be 3 or 4, was{len(u)}")


def travel_time(source, receiver, speed_of_sound):
    """
    Calculate time required for sound to travel from a souce to a receiver

    Args:
        source: cartesian position [x,y] or [x,y,z] of sound source
        receiver: cartesian position [x,y] or [x,y,z] of sound receiver
        speed_of_sound: speed of sound in m/s

    Returns:
        time in seconds for sound to travel from source to receiver
    """
    distance = np.linalg.norm(np.array(source) - np.array(receiver))
    return distance / speed_of_sound


def soundfinder(
    receiver_positions,
    arrival_times,
    temperature=20.0,  # celcius
    invert_alg="gps",  # options: 'gps'
    center=True,  # True for original Sound Finder behavior
    pseudo=True,  # False for original Sound Finder
):

    """
    Perform TDOA localization on a sound event

    Localize a sound event given relative arrival times at multiple receivers.
    This function implements a localization algorithm from the
    equations described in the class handout ("Global Positioning
    Systems"). Localization can be performed in a global coordinate
    system in meters (i.e., UTM), or relative to recorder positions
    in meters.

    Args:
        receiver_positions: a list of [x,y,z] positions for each receiver
          Positions should be in meters, e.g., the UTM coordinate system.

        arrival_times: a list of TDOA times (onset times) for each recorder
          The times should be in seconds.

        temperature: ambient temperature in Celsius

        invert_alg: what inversion algorithm to use (only 'gps' is implemented)

        center: whether to center recorders before computing localization
          result. Computes localization relative to centered plot, then
          translates solution back to original recorder locations.
          (For behavior of original Sound Finder, use True)

        pseudo: whether to use the pseudorange error (True) or
          sum of squares discrepancy (False) to pick the solution to return
          (For behavior of original Sound Finder, use False. However,
          in initial tests, pseudorange error appears to perform better.)

    Returns:
        The solution (x,y,z,b) with the lower sum of squares discrepancy
        b is the error in the pseudorange (distance to mics), b=c*delta_t (delta_t is time error)
    """
    # make sure our inputs follow consistent format
    receiver_positions = np.array(receiver_positions).astype("float64")
    arrival_times = np.array(arrival_times).astype("float64")

    # The number of dimensions in which to perform localization
    dim = receiver_positions.shape[1]

    # Calculate speed of sound
    speed_of_sound = calc_speed_of_sound(temperature)

    ##### Shift coordinate system to center receivers around origin #####
    if center:
        warnings.warn("centering")
        p_mean = np.mean(receiver_positions, 0)
        receiver_positions = np.array([p - p_mean for p in receiver_positions])
    else:
        warnings.warn("not centering")

    ##### Compute B, a, and e #####
    # Find the pseudorange, rho, for each recorder
    # pseudorange (minus a constant) ~= distances from source to each receiver
    rho = np.array([arrival_times * (-1 * speed_of_sound)]).T

    # Concatenate the pseudorange column to form matrix B
    B = np.concatenate((receiver_positions, rho), axis=1)

    # Vector of ones
    e = np.ones(receiver_positions.shape[0])

    # The vector of squared Lorentz norms
    a = 0.5 * np.apply_along_axis(lorentz_ip, axis=1, arr=B)

    # choose between two algorithms to invert the matrix
    if invert_alg == "lstsq":
        raise NotImplementedError
        # Compute B+ * a and B+ * e
        # using closest equivalent to R's solve(qr(B), e)
        # Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
        # Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

    else:  # invert_alg == 'gps' or 'special'
        ## Compute B+ = (B^T \* B)^(-1) \* B^T
        # B^T * B

        to_invert = np.matmul(B.T, B)

        try:
            inverted = np.linalg.inv(to_invert)

        except np.linalg.LinAlgError as err:
            # for 'gps' algorithm, simply fail
            if invert_alg == "gps":
                warnings.warn("4")
                if "Singular matrix" in str(err):
                    warnings.warn("5")
                    warnings.warn(
                        "Singular matrix. Were recorders linear or on same plane? Exiting with NaN outputs",
                        UserWarning,
                    )
                    return [[np.nan]] * (dim)
                else:
                    warnings.warn("6")
                    raise

            # for 'special' algorithm: Fall back to lstsq algorithm
            else:  # invert_alg == 'special'
                warnings.warn("7")
                Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
                Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

        else:  # inversion of the matrix succeeded
            # Compute B+ * a and B+ * e
            Bplus = np.matmul(inverted, B.T)
            Bplus_a = np.matmul(Bplus, a)
            Bplus_e = np.matmul(Bplus, e)

    ###### Solve quadratic equation for lambda #####

    # Compute coefficients
    cA = lorentz_ip(Bplus_e)
    cB = 2 * (lorentz_ip(Bplus_e, Bplus_a) - 1)
    cC = lorentz_ip(Bplus_a)

    # Compute discriminant
    disc = cB**2 - 4 * cA * cC
    # If discriminant is negative, set to zero to ensure
    # we get an answer, albeit not a very good one
    if disc < 0:
        disc = 0
        warnings.warn(
            "Discriminant negative--set to zero. Solution may be inaccurate. Inspect final value of output array",
            UserWarning,
        )

    # Compute options for lambda
    lamb = (-cB + np.array([-1, 1]) * np.sqrt(disc)) / (2 * cA)

    # Find solution u0 and solution u1
    ale0 = np.add(a, lamb[0] * e)
    u0 = np.matmul(Bplus, ale0)
    ale1 = np.add(a, lamb[1] * e)
    u1 = np.matmul(Bplus, ale1)

    # print('Solution 1: {}'.format(u0))
    # print('Solution 2: {}'.format(u1))

    ##### Return the better solution #####

    # Re-translate points
    if center:
        shift = np.append(p_mean, 0)  # 0 for b=error, which we don't need to shift
        u0 += shift
        u1 += shift

    # Select and return quadratic solution
    if pseudo:
        # Return the solution with the lower error in pseudorange
        # (Error in pseudorange is the final value of the position/solution vector)
        if abs(u0[-1]) <= abs(u1[-1]):
            return u0
        else:
            return u1

    else:
        # This was the return method used in the original Sound Finder,
        # but it gives worse performance

        # Compute sum of squares discrepancies for each solution
        s0 = float(np.sum((np.matmul(B, u0) - np.add(a, lamb[0] * e)) ** 2))
        s1 = float(np.sum((np.matmul(B, u1) - np.add(a, lamb[1] * e)) ** 2))

        # Return the solution with lower sum of squares discrepancy
        if s0 < s1:
            return u0
        else:
            return u1


def gillette_localize(
    receivers=list,
    delays=list,
    temp=20,
    m=[0],
    exact=True,
    summary=False,
    confint=False,
    alpha=0.05,
    td_error=False,
    total_td_error=False,
):
    """
    Calculate the estimated location of a sound's source using the
    algorithm laid out in Gillette and Silverman (2008)
    Args:
        receivers: A numpy array of coordinates for microphones used to
        record the sound. The number of microphones needed should
        be two more than the dimensions being localized in. The
        first row will be treated as a reference point for the
        algorithm.
        tdoa: A list of time delays. Each entry should be the time
        delay for the corresponding item in the receivers list
        (i.e. the first item is the delay for the first receiver).
        The first item in this list should be 0, with all other
        entries centered around that.
        temp: ambient temperature in Celsius. Defaults to 20.
        exact: computes an exact solution if True, computes estimates
        with uncertainty if false. Defaults to True
        summary: displays a summary of the estimates if True. Defaults
        to false.
        confint: outputs confidence intervals for the estimated
        coordinates if true. Defaults to false.
        alpha: Determines confidence level of the confidence intervals.
        Defaults to 0.05.
        m: the index of the reference mic. Defaults to 0.
        td_error: Computes the expected time delay from the estimated
        source location, centered around the reference mic, for each
        microphone.
        total_td_error: Computes the euclidean norm of the errors
        provided by td_error.
    Returns:
        an array with the estimated coordinates and the estimated
        distance from the reference mic. (One reference mic and two
        additional mics, this is a 2 item array containing an estima
        -ted x coordinate and a distance.)
    """
    import opensoundscape.localization as loc
    import statsmodels.api as sm

    C = loc.calc_speed_of_sound(temperature=20)
    # Compile know receiver locations and distance delays into an output vector
    out_knowns = []
    in_knowns = np.zeros(((len(receivers) - len(m)) * len(m), 2 + len(m)))
    toa = np.array(delays)
    r = 0
    out_knowns = []
    for k in range(len(m)):
        tdoa = (
            toa - toa[m[k]]
        )  # Use the speed of sound to convert time delays to "distance delays"
        diffs = []
        for delay in tdoa:
            diffs.append(float(delay * loc.calc_speed_of_sound(20)))
        for i in range(len(receivers)):
            if i in m:
                continue
            else:
                w = diffs[i] ** 2
                for j in range(len(receivers[i])):
                    w = w - receivers[i][j] ** 2 + receivers[m[k]][j] ** 2
                w = w / 2
                out_knowns.append(w)
        for i in range(len(receivers)):
            if i in m:
                continue
            else:
                q = 0
                for j in range(len(receivers[i])):
                    z = receivers[m[k]][j] - receivers[i][j]
                    in_knowns[r][q] = z
                    q += 1
                in_knowns[r][q + k] = diffs[i]
                r += 1
                continue

        # Using least squares, compute the final estimated location of source
    location = sm.OLS(out_knowns, in_knowns).fit()
    return (
        location.params,
        location.summary(alpha=alpha),
        location.conf_int(alpha=alpha),
    )
    if summary == True:
        return location.summary()


def calc_tdoa_errors(receivers, tdoas, estimate, temp=20):
    """
    From a set of TDOAs, receivers, and an estimated location,
    calculate the errors in the TDOAs.
    Args:
        - receivers: a list of receiver locations
        - tdoa: a list of time delays
        - estimate: the estimated location of the source
        - temp: ambient temperature in Celsius. Defaults to 20.
        Otherwise, returns the total error in the TDOAs.
    Returns:
        - a list of errors (in seconds) of the TDOAs
    """
    speed_of_sound = calc_speed_of_sound(temperature=temp)
    expected_delays = [
        np.linalg.norm(mic - estimate) / speed_of_sound for mic in receivers
    ]
    delay_errors = [abs(tdoa[i] - expected_delay[i]) for i in range(len(tdoa))]
    return delay_errors
