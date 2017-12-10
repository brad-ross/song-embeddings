import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from itertools import permutations
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import v_measure_score as vm
from sklearn.metrics import homogeneity_score as hom
from sklearn.metrics import completeness_score as com

class TestingGauntlet:

    def __init__(self):
        self.eval_metrics = {
            "Adjusted Mutual Information": ami,
            "Homogeneity": hom,
            "Completeness": com,
            "V-measure" : vm
        }

        self.clustering_tests = {
            "k-means Task" : self.kmeans_test,
            "Mixture of Gaussians Task" : self.mog_test
        }


    def run_tests(self, embedding_fn, specs, labels):
        embedding = embedding_fn(specs)
        test_results = {}
        for test in self.clustering_tests:
            test_results[test] = self.clustering_tests[test](embedding, labels)
        return test_results, emebdding


    def get_scores(self, true_labels, pred_labels):
        scores = {}
        for ev in self.eval_metrics:
            scores[ev] = self.eval_metrics[ev](true_labels, pred_labels)
        return scores

    def kmeans_test(self, embedding, labels, n_genres=None):
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
        return self.get_scores(labels, p_labels)



    def mog_test(self, embedding, labels, n_genres=None):
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
        return self.get_scores(labels, p_labels)



    def print_results(self, results):
        for test in results:
            print "~~~~~~~~~~~~ Results for ", test, "~~~~~~~~~~~~"
            for metric in results[test]:
                print "{:<30}".format(metric + ":"), results[test][metric]
