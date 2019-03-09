"""
Methods for calculating joint parameters, such as joint centers and axes.

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import array, zeros, logical_and, abs as nabs, concatenate, cross, std, sign, argmax
from numpy.linalg import lstsq, norm
from scipy.optimize import least_squares


class Center:
    def __init__(self, g=9.81, method='SAC', mask_input=True, min_samples=1000, opt_kwargs={}):
        """
        Object for joint center computation

        Parameters
        ----------
        g : float, optional
            Local value of gravitational acceleration. Default is 9.81 m/s^2.
        method : {'SAC', 'SSFC', 'SSFCv'}, optional
            Method to use for the computation of the joint center. Default is SAC. See Crabolu et al. for more details.
            SSFCv is SSFC but using vectors instead of magnitude, which requires rotations between sensors.
        mask_input : bool, optional
            Mask the input to only use the highest acceleration samples. Default is True
        min_samples : int, optional
            Minimum number of samples to use. Default is 1000.
        opt_kwargs : dict, optional
            Optimization key-word arguments. SAC uses numpy.linalg.lstsq.
            SSFC and SSFCv use scipy.optimize.least_squares.

        References
        ----------
        Crabolu et al. In vivo estimation of the shoulder joint center of rotation using magneto-inertial sensors:
        MRI-based accuracy and repeatability assessment. BioMedical Engineering Online. 2017.
        """
        self.g = g
        self.method = method
        self.mask_input = mask_input
        self.min_samples = min_samples
        self.opt_kwargs = opt_kwargs

    def compute(self, prox_a, dist_a, prox_w, dist_w, prox_wd, dist_wd, R_dist_prox):
        """
        Perform the computation of the joint center to sensor vectors.

        Parameters
        ----------
        prox_a : numpy.ndarray
            Nx3 array of accelerations measured by the joint proximal sensor.
        dist_a : numpy.ndarray
            Nx3 array of accelerations measured by the joint distal sensor.
        prox_w : numpy.ndarray
            Nx3 array of angular velocities measured by the joint proximal sensor.
        dist_w : numpy.ndarray
            Nx3 array of angular velocities measured by the joint distal sensor.
        prox_wd : numpy.ndarray
            Nx3 array of angular accelerations measured by the joint proximal sensor.
        dist_wd : numpy.ndarray
            Nx3 array of angular accelerations measured by the joint distal sensor.
        R_dist_prox : numpy.ndarray
            Nx3x3 array of rotations from the distal sensor frame to the proximal sensor frame.

        Returns
        -------
        prox_r : numpy.ndarray
            Joint center to proximal sensor vector.
        dist_r : numpy.ndarray
            Joint center to distal sensor vector.
        residual : float
            Residual value per sample used from the joint center optimization
        """
        if self.method == 'SAC':
            if self.mask_input:
                prox_an = norm(prox_a, axis=1) - self.g
                dist_an = norm(dist_a, axis=1) - self.g

                mask = zeros(prox_an.shape, dtype=bool)
                thresh = 0.8
                while mask.sum() < self.min_samples:
                    mask = logical_and(nabs(prox_an) > thresh, nabs(dist_an) > thresh)

                    thresh -= 0.05
                    if thresh < 0.09:
                        raise ValueError('Not enough samples or samples with high motion in the trial provided.  '
                                         'Use another trial')
            else:
                mask = zeros(prox_a.shape[0], dtype=bool)
                mask[:] = True

            # create the skew symmetric matrix products
            prox_K = array([[-prox_w[mask, 1] ** 2 - prox_w[mask, 2] ** 2,
                             prox_w[mask, 0] * prox_w[mask, 1] - prox_wd[mask, 2],
                             prox_wd[mask, 1] + prox_w[mask, 0] * prox_w[mask, 2]],
                            [prox_wd[mask, 2] + prox_w[mask, 0] * prox_w[mask, 1],
                             -prox_w[mask, 0] ** 2 - prox_w[mask, 2] ** 2,
                             prox_w[mask, 1] * prox_w[mask, 2] - prox_wd[mask, 0]],
                            [prox_w[mask, 0] * prox_w[mask, 2] - prox_wd[mask, 1],
                             prox_wd[mask, 0] + prox_w[mask, 1] * prox_w[mask, 2],
                             -prox_w[mask, 0] ** 2 - prox_w[mask, 1] ** 2]]).transpose([2, 0, 1])

            dist_K = array([[-dist_w[mask, 1] ** 2 - dist_w[mask, 2] ** 2,
                             dist_w[mask, 0] * dist_w[mask, 1] - dist_wd[mask, 2],
                             dist_wd[mask, 1] + dist_w[mask, 0] * dist_w[mask, 2]],
                            [dist_wd[mask, 2] + dist_w[mask, 0] * dist_w[mask, 1],
                             -dist_w[mask, 0] ** 2 - dist_w[mask, 2] ** 2,
                             dist_w[mask, 1] * dist_w[mask, 2] - dist_wd[mask, 0]],
                            [dist_w[mask, 0] * dist_w[mask, 2] - dist_wd[mask, 1],
                             dist_wd[mask, 0] + dist_w[mask, 1] * dist_w[mask, 2],
                             -dist_w[mask, 0] ** 2 - dist_w[mask, 1] ** 2]]).transpose([2, 0, 1])

            # create the oversized A and b matrices
            A = concatenate((prox_K, -R_dist_prox[mask] @ dist_K), axis=2).reshape((-1, 6))
            b = (prox_a[mask].reshape((-1, 3, 1))
                 - R_dist_prox[mask] @ dist_a[mask].reshape((-1, 3, 1))).reshape((-1, 1))

            # solve the linear least squares problem
            r, residual, _, _ = lstsq(A, b, rcond=None)
            r.resize((6,))
            residual = residual[0]

        elif self.method == 'SSFC':
            r_init = zeros((6,))

            if self.mask_input:
                prox_an = norm(prox_a, axis=1) - self.g
                dist_an = norm(dist_a, axis=1) - self.g

                mask = zeros(prox_an.shape, dtype=bool)
                thresh = 0.8
                while mask.sum() < self.min_samples:
                    mask = logical_and(nabs(prox_an) > thresh, nabs(dist_an) > thresh)

                    thresh -= 0.05
                    if thresh < 0.09:
                        raise ValueError('Not enough samples or samples with high motion in the trial provided.  '
                                         'Use another trial')
            else:
                mask = zeros(prox_a.shape[0], dtype=bool)
                mask[:] = True

            # create the arguments to be passed to both the residual and jacobian calculation functions
            args = (prox_a[mask], dist_a[mask], prox_w[mask], dist_w[mask], prox_wd[mask], dist_wd[mask])

            sol = least_squares(Center._compute_distance_residuals, r_init.flatten(), args=args, **self.opt_kwargs)
            r = sol.x
            residual = sol.cost

        return r[:3], r[3:], residual / mask.sum()

    @staticmethod
    def _compute_distance_residuals(r, a1, a2, w1, w2, wd1, wd2):
        """
            Compute the residuals for the given joint center locations for proximal and distal inertial data

            Parameters
            ----------
            r : numpy.ndarray
                6x1 array of joint center locations.  First three values are proximal location guess, last three values
                are distal location guess.
            a1 : numpy.ndarray
                Nx3 array of accelerations from the proximal sensor.
            a2 : numpy.ndarray
                Nx3 array of accelerations from the distal sensor.
            w1 : numpy.ndarray
                Nx3 array of angular velocities from the proximal sensor.
            w2 : numpy.ndarray
                Nx3 array of angular velocities from the distal sensor.
            wd1 : numpy.ndarray
                Nx3 array of angular accelerations from the proximal sensor.
            wd2 : numpy.ndarray
                Nx3 array of angular accelerations from the distal sensor.

            Returns
            -------
            e : numpy.ndarray
                Nx1 array of residuals for the given joint center location guess.
            """
        r1 = r[:3]
        r2 = r[3:]

        at1 = a1 - cross(w1, cross(w1, r1, axisb=0)) - cross(wd1, r1, axisb=0)
        at2 = a2 - cross(w2, cross(w2, r2, axisb=0)) - cross(wd2, r2, axisb=0)

        return norm(at1, axis=1) - norm(at2, axis=1)


class KneeAxis:
    def __init__(self, mask_input=True, min_samples=1500, opt_kwargs={}):
        """

        Parameters
        ----------
        mask_input : bool, optional
            Mask the input to only use samples with enough angular velocity to give a good estimate.  Default is True.
        min_samples : int, optional
            Minimum number of samples to use in the optimization.  Default is 1500.
        opt_kwargs : dict, optional
            Optimization key-word arguments.  See scipy.optimize.least_squares.

        References
        ----------
        Seel et al. IMU-Based Joint Angle Measurement for Gait Analysis. Sensors. 2014
        Seel et al. Joint axis and position estimation from inertial measurement data by exploiting kinematic
        constraints. 2012 IEEE International Conference on Control Applications. 2012
        """
        self.mask_input = mask_input
        self.min_samples = min_samples
        self.opt_kwargs = opt_kwargs

    def compute(self, thigh_w, shank_w):
        """
        Compute the knee axis using the given angular velocities.

        Parameters
        ----------
        thigh_w : numpy.ndarray
            Nx3 array of angular velocities measured by the thigh sensor.
        shank_w : numpy.ndarray
            Nx3 array of angular velocities measured by the shank sensor.

        Returns
        -------
        thigh_j : numpy.ndarray
            Vector of the joint rotation axis in the thigh sensor frame.
        shank_j : numpy.ndarray
            Vector of the joint rotation axis in the shank sensor frame.
        """
        j_init = zeros(6)

        if self.mask_input:
            thigh_wn = norm(thigh_w, axis=1)
            shank_wn = norm(shank_w, axis=1)

            factor = 1.0
            mask = zeros(thigh_wn.size, dtype=bool)
            while mask.sum() < self.min_samples:
                mask = (thigh_wn > (factor * std(thigh_wn))) & (shank_wn > (factor * std(shank_wn)))
                factor -= 0.1
                if factor < 0.15:
                    raise ValueError('Not enough samples to mask and still estimate the joint axis.  Consider not '
                                     'masking, or use a trial with more samples')
        else:
            mask = array([True] * thigh_w.shape[0])

            # arguments for the solver
        args = (thigh_w[mask], shank_w[mask])
        # solve and normalize the result
        sol = least_squares(KneeAxis._compute_axis_residuals, j_init.flatten(), args=args, **self.opt_kwargs)
        thigh_j = sol.x[:3] / norm(sol.x[:3])
        shank_j = sol.x[3:] / norm(sol.x[3:])

        return thigh_j, shank_j

    @staticmethod
    def _compute_axis_residuals(j, thigh_w, shank_w):
        """
        Compute the residuals using the estimate of the axis.

        Parameters
        ----------
        j : numpy.ndarray
            Estimates of thigh and shank axis stacked into one vector
        thigh_w : numpy.ndarray
            Nx3 array of angular velocities measured by the thigh sensor.
        shank_w : numpy.ndarray
            Nx3 array of angular velocities measured by the shank sensor.

        Returns
        -------
        e : numpy.ndarray
            N length array of residuals.
        """
        j1 = j[:3] / norm(j[:3])
        j2 = j[3:] / norm(j[3:])

        wp1 = cross(thigh_w, j1)
        wp2 = cross(shank_w, j2)

        return norm(wp1, axis=1) - norm(wp2, axis=1)


def correct_knee(thigh_w, shank_w, thigh_r, shank_r, R_thigh_shank, knee_axis_kwargs={}):
    """
    Correct the knee position based on the computed knee axis.

    Parameters
    ----------
    thigh_w : numpy.ndarray
        Nx3 array of angular velocities measured by the thigh sensor.
    shank_w : numpy.ndarray
        Nx3 array of angular velocities measured by the shank sensor.
    thigh_r : numpy.ndarray
        Initial knee joint center to thigh sensor vector.
    shank_r : numpy.ndarray
        Initial knee joint center to shank sensor vector.
    R_thigh_shank : numpy.ndarray
        Rotation matrix from shank to thigh sensors.
    knee_axis_kwargs : dict, optional
        Knee axis computation key-word arguments. See KneeAxis.

    Returns
    -------
    thigh_r_corr : numpy.ndarray
        Corrected knee joint center to thigh sensor vector.
    shank_r_corr : numpy.ndarray
        Corrected knee joint center to shank sensor vector.
    """
    # compute the knee axis
    ka = KneeAxis(**knee_axis_kwargs)
    thigh_j, shank_j = ka.compute(thigh_w, shank_w)

    # check the sign of the major component of the axes when rotated into the same frame
    shank_j_thigh = R_thigh_shank @ shank_j
    if sign(shank_j_thigh[argmax(nabs(shank_j_thigh))]) != sign(thigh_j[argmax(nabs(thigh_j))]):
        shank_j *= -1

    # compute the corrections for the joint centers
    tmp = (sum(thigh_r * thigh_j) + sum(shank_r * shank_j)) / 2
    thigh_r_corr = thigh_r - thigh_j * tmp
    shank_r_corr = shank_r - shank_j * tmp

    return thigh_r_corr, shank_r_corr


def fixed_axis(center1, center2, center_to_sensor=True):
    """
    Compute the fixed axis of a segment based on computed joint centers.

    Parameters
    ----------
    center1 : numpy.ndarray
        Location of the first joint center. This will be the "origin" of the axis, unless the locations provided
        are vectors from joint center to sensor.
    center2 : numpy.ndarray
        Location of the second joint center. This will be the "end" of the axis, unless the locations provided are
        vectors from joint center to sensor, in which case it will be the "origin".
    center_to_sensor : bool, optional
        If the vectors provided are joint center to sensor (opposite is sensor to joint center). If True, then the
        axes are created in the opposite way as expected (eg right pointing pelvis axis would be left hip joint center
        minus the right hip joint center). Default is True.

    Returns
    -------
    axis : numpy.ndarray
        Fixed axis based on joint centers.
    """

    if center_to_sensor:
        axis = center1 - center2
    else:
        axis = center2 - center1

    # normalize
    axis /= norm(axis)
    return axis

