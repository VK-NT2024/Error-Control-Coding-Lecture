import numpy as np


def bi2de(bi, leftMSB=True):
    """
    Converts an array of binary number(s) where each row contains the binary equivalent is arranged from
        MSB to LSB: if leftMSB is True
        LSB to MSB: if leftMSB is False
        to an array of decimal number
    """
    # check if array and dtype is float or int
    bi = np.array(bi)

    if bi.dtype not in ['float64', 'float32', 'int64', 'int32']:
        raise TypeError('float or int expected')

    if (np.array(np.shape(bi))).size < 2:
        bi = np.reshape(bi, (1, np.shape(bi)[0]))

    # check if first column is all zeros and delete
    if np.all(bi[:, 0] == 0):
        bi = np.delete(bi, 0, axis=1)

    K, N = bi.shape
    if leftMSB:
        powers = np.flip(2 ** np.arange(0, N))
    else:
        powers = 2 ** np.arange(0, N)
    dec = np.sum((bi * powers), axis=1, dtype=int)
    return dec
