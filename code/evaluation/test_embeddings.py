from testing_rig import TestingGauntlet
from ..data.cloud_storage import open_file_in_bucket, get_path_to_file_in_bucket
from ..data.dataset_utils import get_numerical_labels
import numpy as np
from ..embeddings.pca.pca_embed import pca_embedding
from ..embeddings.raw.raw_embed import raw_embedding
from ..embeddings.ali.ali_embed_l2 import create_embedding_fn
from ..embeddings.ali.ali_model_l2 import ALIModel

from plot_embeddings import *
from bar_plotting import *

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

GENRE_TO_LABEL = {
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

GENRE_ORDER =  [
    'classical',
    'jazz',
    'latin',
    'folk',
    
    'country',
    'funk',
    'indie rock',
    'rock',
    'metal',   
 
    'pop', 
    'r&b',
    'hip hop',
    'rap',
 
    'house',
    'edm',
]

#things that are common that don't really need to be changed
#(it was annoying to pass these around EVERYWHERE)
ali_embedding = create_embedding_fn(ALIModel, 'model_weights_8_epoch_4')
label_mapping, labels = get_numerical_labels(get_path_to_file_in_bucket('real_labels.csv', 'song-embeddings-dataset'))[108000:120000]
specs = np.load(get_path_to_file_in_bucket('108000_120000.npy', 'song-embeddings-dataset'))

def perform_tests(embedding_fn, genres, num_per_genre, tsne_params, test_name, plot_type='t-sne', fn_name="", plot_path=None):
    chosen_genres = [GENRE_TO_LABEL[g] for g in genres]
    positions = np.array([], dtype=int)
    for cg in chosen_genres:
        #gets first num_per_genre of genre cg. alternatively, we can get random ones
        positions = np.append(positions, np.where(labels == cg)[0][:num_per_genre])
    new_specs = specs[positions]
    new_labels = labels[positions]

    gauntlet = TestingGauntlet()
    results, embedding = gauntlet.run_tests(embedding_fn, new_specs, new_labels)
    print fn_name, 'results: '
    gauntlet.print_results(results)
    if plot_path != None:
        plot_embedding(embedding, new_labels, \
                       title=test_name + ': Embedding using ' + fn_name,\
		       plot_type=plot_type,
		       tsne_params=tsne_params,
                       save_path=plot_path + '_embed_plot.png',\
                       label_dict=LABEL_TO_GENRE,
                       label_order=GENRE_ORDER)
    return results

def run_comparison_tests(genres, num_per_genre, emebdding_fns=[pca_embedding, ali_embedding],\
                         fn_names=['PCA Embedding', 'ALI Embedding'],\
                         test_name='',
			 plot_type='t-sne',
			 tsne_params={},
                         plot_path=None):
    if plot_path != None:
        paths = [plot_path + '_pca', plot_path + '_ali']
    else:
        paths = [None, None]
    
    results_pca = perform_tests(emebdding_fns[0], genres, num_per_genre,\
                                tsne_params, plot_type=plot_type, test_name=test_name, fn_name=fn_names[0], plot_path=paths[0])
    results_ali = perform_tests(emebdding_fns[1], genres, num_per_genre,\
                                tsne_params, plot_type=plot_type, test_name=test_name, fn_name=fn_names[1], plot_path=paths[1])

    if plot_path != None:
        make_bar_plot(results_pca, results_ali, ['Adjusted Mutual Information', 'V-measure'],\
                      fn_names, plot_path + '_bar', test_name=test_name)


def do_all_tests_make_all_plots(genre_sets, nums_per_genre, test_names, plot_types, tsne_params, plot_path=None):
    for i in range(len(genre_sets)):
        genres_to_run = genre_sets[i]
        num_per_genre = nums_per_genre[i]
        test_name = test_names[i]
        plot_type = plot_types[i]
	tsne_param = tsne_params[i]

        curr_plot_path = plot_path
        if plot_path != None:
            curr_plot_path += '_' + string_to_filename(test_name)

        print '===== ' + test_name + ' ====='
        run_comparison_tests(genres_to_run, num_per_genre, test_name=test_name, plot_type=plot_type, tsne_params=tsne_param, plot_path=curr_plot_path)
        print '\n'

def string_to_filename(s):
    return '_'.join([c.lower() for c in s.split(' ')])

test_names = [
    #'Four Genres',
    #'Jazz + Rock ?= Funk',
    'All Genres'
]

genre_sets = [
   # ['rock', 'indie rock', 'edm', 'classical'],
   # ['jazz', 'rock', 'funk'],
    GENRE_TO_LABEL.keys()
]

plot_types = [
    #'t-sne',
   # 't-sne',
    't-sne'
]

nums_per_genre = [
  #  50,
 #   50,
    50
]

tsne_params = [
#{'learning_rate': 50, 'perplexity': 20, 'n_iter': 1000},
#{'learning_rate': 50, 'perplexity': 20, 'n_iter': 1000},
{'perplexity': 28, 'n_iter': 1000, 'learning_rate': 50}
]

save_path     = get_path_to_file_in_bucket('embedding_plot', 'song-embeddings-dataset')

do_all_tests_make_all_plots(genre_sets, nums_per_genre, test_names, plot_types, tsne_params, plot_path=save_path)

#just one test:
# genres_to_run = ['rock', 'indie rock', 'edm', 'classical']
# num_per_genre = 50
# save_path     = get_path_to_file_in_bucket('plotting_test', 'song-embeddings-dataset')
#
# run_comparison_tests(specs, labels, genres_to_run, num_per_genre, plot_path=save_path)

#
# embedding_fn  = pca_embedding
# fn_name       = "PCA Embedding"
# perform_tests(embedding_fn, specs, labels, genres_to_run, num_per_genre,\
#               fn_name=fn_name, plot=True, plot_path=save_path)
#
# genres_to_run = ['rock', 'indie rock', 'edm', 'classical']
# num_per_genre = 50
# embedding_fn  = create_embedding_fn(ALIModel, 'model_weights_8_epoch_4')
# fn_name       = "ALI Embedding"
# save_path     = get_path_to_file_in_bucket('ali_embedding_test_plot.png', 'song-embeddings-dataset')
# perform_tests(embedding_fn, specs, labels, genres_to_run, num_per_genre,\
#               fn_name=fn_name, plot=True, plot_path=save_path)
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
