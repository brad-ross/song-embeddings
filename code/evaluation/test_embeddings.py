from testing_rig import TestingGauntlet
from ..data.cloud_storage import open_file_in_bucket, get_path_to_file_in_bucket
from ..data.dataset_utils import get_numerical_labels
import numpy as np
from ..embeddings.pca.pca_embed import pca_embedding
from ..embeddings.raw.raw_embed import raw_embedding
from ..embeddings.ali.ali_embed_l2 import create_embedding_fn
from ..embeddings.ali.ali_model_l2 import ALIModel

from plot_embeddings import *


LABEL_TO_GENRE = {
    0: 'classical',
    1: 'country',
    2: 'edm',
    3: 'folk',
    4: 'funk',
    5: 'hip hop',
    6: 'house',
    7: 'indie rock',
    8: 'jazz',
    9: 'latin',
    10: 'metal',
    11: 'pop',
    12: 'r&b',
    13: 'rap',
    14: 'rock'
}

GENRE_TO_LABEL = = {
    'latin': 9,
    'indie rock': 7,
    'classical': 0,
    'country': 1,
    'rock': 14,
    'jazz': 8,
    'metal': 10,
    'folk': 3,
    'edm': 2,
    'r&b': 12,
    'pop': 11,
    'hip hop': 5,
    'rap': 13,
    'house': 6,
    'funk': 4
}


def perform_tests(emebdding_fn, specs, labels, genres, num_per_genre,
                fn_name="", plot=False, plot_path=None):
    chosen_genres = [GENRE_TO_LABEL[g] for g in genres]
    positions = np.array([])
    for cg in chosen_genres:
        #gets first num_per_genre of genre cg. alternatively, we can get random ones
        positions = np.append(positions, np.where(labels == cg)[0][:num_per_genre])
    new_specs = specs[positions]
    new_labels = labels[positions]

    results, embedding = gauntlet.run_tests(embedding_fn, specs, labels)
    print fn_name, 'results: '
    gauntlet.print_results(results)
    if plot:
        plot_embedding(embedding, new_labels, \
                       title='Embedding using ' + fn_name,\
                       save_path=plot_path,\
                       label_dict=LABEL_TO_GENRE)


labels = get_numerical_labels(get_path_to_file_in_bucket('real_labels.csv', 'song-embeddings-dataset'))[108000:120000]
specs = np.load(get_path_to_file_in_bucket('108000_120000.npy', 'song-embeddings-dataset'))


genres_to_run = ['classical', 'edm', 'rock']
num_per_genre = 20
embedding_fn  = pca_embedding
fn_name       = "PCA embedding"
perform_tests(embedding_fn, specs, labels, genres_to_run, num_per_genre,\
              fn_name=fn_name, plot=True)


#
# small_indices = np.where((labels == 2) | (labels == 0) | (labels == 14))
# labels = labels[small_indices]
# specs = specs[small_indices]
# print(np.unique(labels, return_counts=True))
#
# gauntlet = TestingGauntlet()
# #raw_results = gauntlet.run_tests(raw_embedding, specs, labels)
# pca_results = gauntlet.run_tests(pca_embedding, specs, labels)
# ali_results = gauntlet.run_tests(create_embedding_fn(ALIModel, 'model_weights_7_epoch_3'), specs, labels)
#
# print('PCA Results:')
# gauntlet.print_results(pca_results)
# print('\n')
# print('ALI Results:')
# gauntlet.print_results(ali_results)
