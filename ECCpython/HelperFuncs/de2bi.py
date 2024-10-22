import numpy as np

def de2bi(dec, m, leftMSB=True):
    """
    Converts a single/array of decimal number(s) into a matrix
    where each row contain the binary equivalent and column is
    arranged from
        MSB to LSB: if leftMSB is True
        LSB to MSB: if leftMSB is False
    """
    # check if array and dtype is float or int
    dec = np.array(dec)

    column_length = m
    row_length = len(dec)
    bin_mat = np.zeros((row_length, column_length),dtype=int)

    for k in range(0, row_length):
        s_bin = np.binary_repr(dec[k], column_length)
        for l in range(0, column_length):
            bin_mat[k][l] = int(s_bin[l])
    if leftMSB == False:
        bin_mat = np.fliplr(bin_mat)

    return bin_mat