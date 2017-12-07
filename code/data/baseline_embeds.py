import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from itertools import permutations
from sklearn.mixture import GaussianMixture


#BASELINE EMBEDDINGS
def raw_spec_embedding(specs):
    """
    Baseline embedding: flatten spectograms and stack them.

    Benefits: simple, easy
    Worries: SUPER SLOW (big matrixes!!)

    """
    return np.array([m.flatten('F') for m in specs]) #F -> flatten along columns, which make more sense

def pca_embedding(specs, r=150):
    """
    Baseline embedding: take the spectograms, flatten them, stack them, and project onto the
    first 150 PCA components.

    Benefits: Faster than raw embedding!!
    Worries: Naive, off the shelf

    """
    X = np.array([m.flatten('F') for m in specs]) #F -> flatten along columns, which make more sense
    X =  X - sum([X[i:i+1] for i in range(X.shape[0])])/float(X.shape[0]) #normalize rows
    pca_ = PCA(n_components=r)
    pca_.fit(X)
    comps = pca_.components_
    # print X.shape, comps.shape #Note: dimension of embeddings will be min(r, # of examples)
    return X.dot(comps.T)





#EVALUATION OF EMBEDDINGS
def eval_wperm(true_labels, pred_labels, perm):
    """
    Helper for eval_labels
    """
    N = len(true_labels)
    count = 0.0
    for i in range(N):
        if true_labels[i] == perm[pred_labels[i]]:
            count += 1
    return count/N


def eval_labels(true_labels, pred_labels):
    """
    One possible way of computing k-means performance:
    iterate over each possibility of what the predicicted sets correspond to, choose
    the one that maximizes success
    """

    poss_labels = set(true_labels)
    return max([eval_wperm(true_labels, pred_labels, perm) for perm in permutations(poss_labels)])


#     N = len(true_labels)
#     count = 0.0
#     for i in range(N):
#         for j in range(N):
#             if true_labels[i] == true_labels[j] and pred_labels[i] == pred_labels[j]:
#                     count += 1
#             if true_labels[i] != true_labels[j] and pred_labels[i] != pred_labels[j]:
#                     count += 1
#     return count/N**2



def kmeans_eval(embedding, labels, n_genres=None):
    """
    Evaluates how well the given embedding performs on the kmeans task

    @param embedding: a list of vectors [v1, v2, ..., vn] of song embeddings
    in R^k.
    @param labels: a list of genres where labels[i] is the genre of embedding[i].
    @param n_genres: the number of genres (for convenience). computed manually if None
    is given.
    """
    if n_genres == None:
        n_genres = len(set(labels))

    p_labels = KMeans(n_clusters=n_genres).fit_predict(embedding)
    #for visual comparison
    # print p_labels
    # print labels
    return eval_labels(labels, p_labels)

def mog_eval(embedding, labels, n_genres=None):
    """
    Evaluates how well the given embedding performs on the mixture of gaussians task

    @param embedding: a list of vectors [v1, v2, ..., vn] of song embeddings
    in R^k.
    @param labels: a list of genres where labels[i] is the genre of embedding[i].
    @param n_genres: the number of genres (for convenience). computed manually if None
    is given.
    """
    if n_genres == None:
        n_genres = len(set(labels))

    clf = GaussianMixture(n_components=n_genres)
    clf.fit(embedding)
    p_labels = clf.predict(embedding)
    #for visual comparison
    # print p_labels
    # print labels
    return p_labels, eval_labels(labels, p_labels)



genres = np.load("genres_5s.npy")
specs = np.load("spectrograms_5s.npy")

oe_specs = specs[range(150) + range(300, 450),:]
oe_grs = genres[range(150) + range(300, 450)]

# rs = [1, 2, 3, 4, 5, 10, 20, 40, 80, 160]
# performances = []
# for r in rs:
big_pca_embed = pca_embedding(oe_specs, r=5)
perf = mog_eval(big_pca_embed, oe_grs)
print perf
#     performances.append(perf)
#     print r, perf
# print performances
