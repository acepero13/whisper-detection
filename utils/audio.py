import numpy as np


def normalize(sig, rms_level=0):
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    # read input file

    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(sig) * r ** 2) / np.sum(sig ** 2))

    # normalize
    y = sig * a

    return y
