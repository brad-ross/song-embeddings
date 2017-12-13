from ..data.cloud_storage import open_file_in_bucket, get_path_to_file_in_bucket
from ..data.dataset_utils import get_numerical_labels
import numpy as np
from ..embeddings.pca.pca_embed import pca_embedding
from ..embeddings.ali.ali_embed_l2 import create_embedding_fn
from ..embeddings.ali.ali_model_l2 import ALIModel

from plot_embeddings import *

def get_positions_of_wanted_labels(labels, wanted_labels, inv_mapping):
    positions = np.array([], dtype=int)
    for l in [inv_mapping[wl] for wl in wanted_labels]:
        positions = np.append(positions, np.where(labels == l)[0])
    return positions

ali_embedding = create_embedding_fn(ALIModel, 'model_weights_8_epoch_4')

def plot_coldplay():
    albums_to_plot = ['Something Just Like This', 'A Head Full Of Dreams', 'Ghost Stories', 'Mylo Xyloto', 
                      'Viva La Vida Or Death And All His Friends', 'X & Y', 'A Rush Of Blood To The Head', 'Parachutes']
    label_mapping, inv_mapping, labels = get_numerical_labels(get_path_to_file_in_bucket('coldplay_labels.csv', 'song-embeddings-artist-experiments'), 3)
    specs = np.load(get_path_to_file_in_bucket('coldplay_0_1170.npy', 'song-embeddings-artist-experiments'))
    positions = get_positions_of_wanted_labels(labels, albums_to_plot, inv_mapping)
    
    specs_embedded_ali = ali_embedding(specs[positions])

    tsne_params = {}
    plot_embedding(specs_embedded_ali, labels[positions], title='Coldplay Songs by Album: Embedded Using ALI', plot_type='pca', tsne_params=tsne_params, \
                   save_path=get_path_to_file_in_bucket('coldplay_ali_embedding_plot_pca_viz', 'song-embeddings-artist-experiments'), 
                   label_dict=label_mapping, label_order=albums_to_plot, legend_outside=True)

def plot_billyjoel():
    #albums_to_plot = ['Something Just Like This', 'A Head Full Of Dreams', 'Ghost Stories', 'Mylo Xyloto', 
    #                  'Viva La Vida Or Death And All His Friends', 'X & Y', 'A Rush Of Blood To The Head', 'Parachutes']
    label_mapping, inv_mapping, labels = get_numerical_labels(get_path_to_file_in_bucket('billyjoel_labels.csv', 'song-embeddings-artist-experiments'), 3)
    specs = np.load(get_path_to_file_in_bucket('billyjoel_0_2690.npy', 'song-embeddings-artist-experiments'))
    #positions = get_positions_of_wanted_labels(labels, albums_to_plot, inv_mapping)
    
    specs_embedded_ali = ali_embedding(specs)

    tsne_params = {}
    plot_embedding(specs_embedded_ali, labels, title='Billy Joel Songs by Album: Embedded Using ALI', plot_type='t-sne', tsne_params=tsne_params, \
                   save_path=get_path_to_file_in_bucket('billyjoel_ali_embedding_plot', 'song-embeddings-artist-experiments'), 
                   label_dict=label_mapping, legend_outside=True)

#plot_coldplay()
plot_billyjoel()
