from testing_rig import TestingGauntlet
from ..data.cloud_storage import open_file_in_bucket, get_path_to_file_in_bucket
from ..data.dataset_utils import get_numerical_labels
import numpy as np
from ..embeddings.pca.pca_embed import pca_embedding
from ..embeddings.raw.raw_embed import raw_embedding
from ..embeddings.ali.ali_embed_l2 import create_embedding_fn
from ..embeddings.ali.ali_model_l2 import ALIModel

labels = get_numerical_labels(get_path_to_file_in_bucket('real_labels.csv', 'song-embeddings-dataset'))[144000:146680]
specs = np.load(get_path_to_file_in_bucket('144000_146680.npy', 'song-embeddings-dataset'))

small_indices = np.where((labels == 2) | (labels == 0) | (labels == 14))
labels = labels[small_indices]
specs = specs[small_indices]
print(np.unique(labels, return_counts=True))

gauntlet = TestingGauntlet()
#raw_results = gauntlet.run_tests(raw_embedding, specs, labels)
pca_results = gauntlet.run_tests(pca_embedding, specs, labels)
ali_results = gauntlet.run_tests(create_embedding_fn(ALIModel, 'model_weights_7_epoch_3'), specs, labels)

print('PCA Results:')
gauntlet.print_results(pca_results)
print('\n')
print('ALI Results:')
gauntlet.print_results(ali_results)
