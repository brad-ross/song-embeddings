
import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


#just for convenience; useful for storage
def get_svd(M):
    return linalg.svd(M)


def low_rank_approx(M, r=150):
#     from scipy import linalg
    """
    Computes an r-rank approx of matrix
    """
    #older, less efficient implementation
    # if svd == None:
    #     svd = linalg.svd(M)
    # u, s, v = svd
    # return u[:,:r].dot(np.diag(s[:r])).dot(v[:r,:])

    U, S, Vt = randomized_svd(M, n_components=r)
    return U.dot(np.diag(S)).dot(Vt)


def shrink_and_flatten(M, r=150):
    """
    Concatenates flattenings of first r right/left sigular vectors
    Reduces size from m x n to m x r, n x r (mn to r(m + n))
    """

    U, S, Vt = randomized_svd(M, n_components=r)
    return np.append(U.T.flatten(), Vt.flatten())



def mat_k_means(matrixes, n_clusters, reduce_=True, dim=150):
    """
    Does k-means clustering on an array of matrixes
    If reduce_, then does dimension reduction, and then flattens, otherwise, just flattens
    Note: dimension reduction makes matrix simpler, but not any smaller
    (see next fn for smaller)
    """
    N_ex = len(matrixes)
    #matrixes with dimesnsions reduced
    if reduce_:
        mats_ = [low_rank_approx(m, r=dim).flatten() for m in matrixes]
    else:
        mats_ = [m.flatten() for m in matrixes]

    #array of predicted clusters
    pred_cls = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(mats_)

    #matrix:predicted cluster
    return [mats_[i]:pred_cls[i] for i in range(N_ex)]



def mat_k_means2(matrixes, n_clusters, reduce=True, dim=150):
    """
    Does k-means clustering on an array of matrixes
    If reduce_, then does dimension reduction as in shrink_and_flatten
    """
    N_ex = len(matrixes)
    #matrixes with dimesnsions reduced
    if reduce_:
        mats_ = [shrink_and_flatten(m) for m in matrixes]
    else:
        mats_ = [m.flatten() for m in matrixes]

    #array of predicted clusters
    pred_cls = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(mats_)

    #matrix:predicted cluster
    return [mats_[i]:pred_cls[i] for i in range(N_ex)]
