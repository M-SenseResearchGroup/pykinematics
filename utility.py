"""
General utility methods used in the calculation of joint angles from inertial sensors

GNU GPL v3.0
Lukas Adamowicz

V0.1 - March 8, 2019
"""
from numpy import array, argmin, zeros, ceil, mean, std, sqrt


def find_most_still(data, window_size, return_index=False):
    """
    Find the most still window of given data by using the minimum of the summed variances for each window.  Returns
    the mean data during the most still window.

    Parameters
    ----------
    data : tuple
        Tuple of data to use for determining the minimum summed variance for each window
    window_size : int
        Number of samples per window.
    return_index : bool
        Return the index of the lowest variance.

    Returns
    -------
    still_data : tuple
        Tuple containing the mean data for each provided data stream for the most still window.
    ind : int, optional
        Index of the lowest variance window, excluding the padding at the start and end of the moving windows.
    """
    var = []
    means = tuple()
    still_data = tuple()
    for d in data:
        m_mn, m_st, pad = mov_avg(d, window_size)

        m_st = m_st ** 2  # square the standard deviation to get the variance

        means += (m_mn[pad:-pad], )
        var.append(m_st[pad:-pad].sum(axis=1))

    ind = argmin(array(var).sum(axis=0))

    for mn in means:
        still_data += (mn[ind], )

    if return_index:
        return still_data, ind
    else:
        return still_data


def calc_derivative(x, dt, order=2):
    """
    Calculate the 2nd or 4th order derivative of a sequence.

    Parameters
    ----------
    x : numpy.ndarray
        1 or 2D array to take the derivative of.
    dt : float
        Time difference for all points.  This cannot handle varying time differences.
    order : {4, 2}, optional
        Order of the derivative to calculate.  Default is 2nd order.

    Returns
    -------
    dx : numpy.ndarray
        Derivative of x.
    """
    dx = zeros(x.shape)

    if order == 4:
        dx[2:-2] = (x[:-4] - 8 * x[1:-3] + 8 * x[3:-1] + x[4:]) / (12 * dt)

        # edges
        dx[:2] = (-25 * x[:2] + 48 * x[1:3] - 36 * x[2:4] + 16 * x[3:5] - 3 * x[4:6]) / (12 * dt)
        dx[-2:] = (25 * x[-2:] - 48 * x[-3:-1] + 36 * x[-4:-2] - 16 * x[-5:-3] + 3 * x[-6:-4]) / (12 * dt)
    elif order == 2:
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)

        # edges
        dx[0] = (-3 * x[0] + 4 * x[1] - x[2]) / (2 * dt)
        dx[-1] = (3 * x[-1] - 4 * x[-2] + x[-3]) / (2 * dt)

    return dx


def mov_avg(seq, window):
    """
    Compute the centered moving average and standard deviation

    Parameters
    ----------
    seq : numpy.ndarray
        Data to take the moving average and standard deviation on.
    window : int
        Window size for the moving average/standard deviation.

    Returns
    -------
    m_mn : numpy.ndarray
        Moving average
    m_st : numpy.ndarray
        Moving standard deviation
    pad : int
        Padding at beginning of the moving average and standard deviation
    """
    m_mn = zeros(seq.shape)
    m_st = zeros(seq.shape)

    if window < 2:
        window = 2

    pad = int(ceil(window / 2))

    # compute first window stats
    m_mn[pad] = mean(seq[:window], axis=0)
    m_st[pad] = std(seq[:window], axis=0, ddof=1)**2  # ddof of 1 indicates sample standard deviation, no population

    # compute moving mean and standard deviation
    for i in range(1, seq.shape[0] - window):
        diff_fl = seq[window + i - 1] - seq[i - 1]  # difference in first and last elements of sliding window
        m_mn[pad + i] = m_mn[pad + i - 1] + diff_fl / window
        m_st[pad + i] = m_st[pad + i - 1] + (seq[window + i - 1] - m_mn[pad + i] + seq[i - 1] -
                                             m_mn[pad + i - 1]) * diff_fl / (window - 1)

    m_st = sqrt(abs(m_st))  # added absolute value to ensure that any round off error doesn't effect the results

    if window % 2 == 1:
        # m_mn = m_mn[:-1]
        # m_st = m_st[:-1]
        pass
    return m_mn, m_st, pad


def quat_mult(q1, q2):
    """
    Multiply quaternions

    Parameters
    ----------
    q1 : numpy.ndarray
        1x4 array representing a quaternion
    q2 : numpy.ndarray
        1x4 array representing a quaternion

    Returns
    -------
    q : numpy.ndarray
        1x4 quaternion product of q1*q2
    """
    if q1.shape != (1, 4) and q1.shape != (4, 1) and q1.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q1 has more or less than 4 elements')
    if q2.shape != (1, 4) and q2.shape != (4, 1) and q2.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q2 has more or less than 4 elements')
    if q1.shape == (4, 1):
        q1 = q1.T

    Q = array([[q2[0], q2[1], q2[2], q2[3]],
               [-q2[1], q2[0], -q2[3], q2[2]],
               [-q2[2], q2[3], q2[0], -q2[1]],
               [-q2[3], -q2[2], q2[1], q2[0]]])

    return q1 @ Q

def quat_conj(q):
    """
    Compute the conjugate of a quaternion

    Parameters
    ----------
    q : numpy.ndarray
        Nx4 array of N quaternions to compute the conjugate of.

    Returns
    -------
    q_conj : numpy.ndarray
        Nx4 array of N quaternion conjugats of q.
    """
    return q * array([1, -1, -1, -1])
