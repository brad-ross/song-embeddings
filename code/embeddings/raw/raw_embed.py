import numpy as np

def raw_spec_embedding(specs):
    """
    Baseline embedding: flatten spectograms and stack them.

    Benefits: simple, easy
    Worries: SUPER SLOW (big matrixes!!)

    """
    return np.reshape(specs, (specs.shape[0], -1))
