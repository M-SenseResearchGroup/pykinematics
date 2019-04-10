"""
Methods that are common and needed accross both IMU and OMC methods

GNU GPL v3.0
Lukas Adamowicz
V0.1 - April 10, 2019
"""
from numpy import array, zeros, argmin, ceil, mean, std, sqrt


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


