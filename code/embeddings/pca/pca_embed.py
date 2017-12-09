import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def pca_embedding(specs, r=64):
    """
    Baseline embedding: take the spectograms, flatten them, stack them, and project onto the
    first 150 PCA components.

    Benefits: Faster than raw embedding!!
    Worries: Naive, off the shelf

    """
    X = scale(np.reshape(specs, (specs.shape[0], -1)))
    pca = PCA(n_components=r)
    pca.fit(X)
    comps = pca.components_
    # print X.shape, comps.shape #Note: dimension of embeddings will be min(r, # of examples)
    return X.dot(comps.T)
