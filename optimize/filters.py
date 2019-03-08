"""
Optimization filters

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import zeros
from numpy.linalg import inv, cholesky


class UnscentedKalmanFilter:
    def __init__(self, x0, P0, F, H, Q, R):
        """
        Parameters
        ----------
        x0 : matrix_like, array_like
            Initial guess for state column vector (Nx1).
        P0 : matrix_like, array_like
            Initial guess for state covariance.  Must be NxN size.
        F : callable
            Function to time-step state vector.  x(t+dt) = F[x(t)]
        H : callable
            Function to estimate measured vector.  z(t) = H[x(t);z(t)]
        Q : matrix_like,array_like,float,optional
            State covariance.  Must be None, NxN matrix/array, or float.  If None,
            defaults to 1.5*I where I is the NxN identity matrix
        R : matrix_like,array_like,float,str,optional
            Process covariance.  Must be None, NxN matrix/array, or float.
            If None, defaults to 1.
        """

        if x0.ndim == 1:
            self.x = x0.reshape((-1, 1))
        else:
            self.x = x0

        self.P = P0
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

    def run1(self, z, f_kwargs={}, h_kwargs={}, sigma_kwargs={}):
        """
        Run Unscented Kalman filter for 1 step

        Parameters
        ----------
        z : array_like
            mx1 array of measured data
        f_kwargs : dict,optional
            Any additional keyword arguments to be passed to state update function
            F.  Optional, defaults to no arguments passed.
        h_kwargs : dict,optional
            Any additional keyword arguments to be passed to measured input estimation
            function H.  Optional, defaults to no arguments passed.
        sigma_kwargs : dict, optional
            Sigma point computation key-word arguments.

        Returns
        ----------------
        x : array_like
            Estimated state variable
        """
        # make sure that z is 2D for proper broadcasting
        if z.ndim == 1:
            z.resize((z.size, 1))

        # compute the sigma points from the state and state covariance
        Xi, W = UnscentedKalmanFilter.sigma_points(self.x, self.P, **sigma_kwargs)

        # update the sigma points
        fXi = self.F(Xi, **f_kwargs)

        # transform sigma points back to normal space
        xp, Px = UnscentedKalmanFilter.unscented_transform(fXi, W, self.Q)

        # get the measurement prediction from the sigma points
        hXi = self.H(fXi, **h_kwargs)

        # transform the measurement prediction to normal space
        zp, Pz = UnscentedKalmanFilter.unscented_transform(hXi, W, self.R)

        Pxz = zeros((self.x.shape[0], z.shape[0]))
        for k in range(2 * self.x.shape[0] + 1):
            Pxz += W[k] * (fXi[:, k].reshape((-1, 1)) - xp) @ (hXi[:, k].reshape((-1, 1)) - zp).T

        # calculate the Kalman gain
        K = Pxz @ inv(Pz)

        # get the best estimate of the state and state covariance
        self.x = xp + K @ (z - zp)
        self.P = Px - K @ Pz @ K.T

        return self.x

    @staticmethod
    def sigma_points(x, P, alpha=0.001, kappa=3.0, beta=2):
        """
        Calculate Sigma Points

        Parameters
        ----------
        x : array_like
            Vector to calculate sigma points for.
        P : array_like
            Covariance array used in sigma point calculation.
        alpha : float, optional
            Alpha value used in sigma point calculation.  Defaults to 0.001.
        kappa : float, optional
            Kappa value used in sigma point calculation.  Defaults to 0.0.
        beta : float, optional
            Beta value, based on the distribution of whats being estimated.

        Returns
        -------
        xi : array
            Sigma points
        W : array
            Weights
        """
        n = x.size
        Xi = zeros((n, 2 * n + 1))
        W = zeros((2 * n + 1, 1))

        # wiki also has different weights for the state and covariance
        # lambda in wiki, kappa in book
        y = alpha ** 2 * (n + kappa) - n

        Xi[:, 0] = x.flatten().copy()
        W[0] = y / (n + y)

        U = cholesky((n + y) * P)

        for i in range(n):
            Xi[:, i + 1] = x.flatten() + U[i, :]
            Xi[:, i + n + 1] = x.flatten() - U[i, :]

        W[1:] = 1 / (2 * (n + y))

        return Xi, W

    @staticmethod
    def unscented_transform(Xi, W, noise_cov=0.0):
        """
        Unscented transform

        Parameters
        ----------
        Xi : array_like
            Sigma Points
        W : array_like
            Weights for sigma points
        noise_cov : float,optional
            Noise covariance.  Optional, defaults to 0.0

        Returns
        -------
        xm : array_like
            Mean of input sigma points from unscented transform
        xcov : array_like
            Covariance of input sigma points from unscented transform
        """
        n, kmax = Xi.shape

        x = Xi @ W

        P = zeros((n, n))
        for k in range(kmax):
            P += W[k] * (Xi[:, k].reshape((-1, 1)) - x) @ (Xi[:, k].reshape((-1, 1)) - x).T

        P += noise_cov

        return x, P


