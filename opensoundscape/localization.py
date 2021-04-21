import numpy as np
import warnings


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


def localize(
    receiver_positions,
    arrival_times,
    temperature=20.0,  # celcius
    invert_alg="gps",  # options: 'lstsq', 'gps'
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

        invert_alg: what inversion algorithm to use

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
        # Compute B+ * a and B+ * e
        # using closest equivalent to R's solve(qr(B), e)
        Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
        Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

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
    disc = cB ** 2 - 4 * cA * cC
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
