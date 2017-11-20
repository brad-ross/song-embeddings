from ..data.cloud_storage import get_path_to_bucket, list_files_in_bucket, open_file_in_bucket
from ..data.audio import mp3_to_array, get_spectrogram
from random import choice, sample
import os
from multiprocessing import Pool
import numpy as np

def get_mp3_filenames(files):
    return filter(lambda f: '.mp3' in f, files)

edm_files = get_mp3_filenames(list_files_in_bucket('song-embeddings-edm'))
rock_files = get_mp3_filenames(list_files_in_bucket('song-embeddings-rock'))
orch_files = get_mp3_filenames(list_files_in_bucket('song-embeddings-orchestral'))

print('Retrieved file lists...')

sample_size = 100

def sample_nonempty_files(files, bucket, k):
    samples = []
    remaining_files = set(files)
    for i in range(k):
        while True:
            f = choice(list(remaining_files))
	    if os.path.getsize(os.path.join(get_path_to_bucket(bucket), f)) == 0:
	        continue
	    remaining_files.remove(f)
	    samples.append(f)
            break
    
    return samples

edm_samples = sample_nonempty_files(edm_files, 'song-embeddings-edm', sample_size)
rock_samples = sample_nonempty_files(rock_files, 'song-embeddings-rock', sample_size)
orch_samples = sample_nonempty_files(orch_files, 'song-embeddings-orchestral', sample_size)

print('Retrieved file samples...')

def get_flat_spectrogram_for(file_tup):
    f, bucket = file_tup
    with open_file_in_bucket(f, bucket) as mp3:
        mp3_array, rate = mp3_to_array(mp3)
        freqs, times, spec = get_spectrogram(mp3_array, rate)
	return spec.flatten()

def get_flat_specs_from(files, bucket): 
    file_bucket_pairs = zip(files, [bucket] * len(files))
    pool = Pool(processes=8)
    specs = pool.map(get_flat_spectrogram_for, file_bucket_pairs)
    return np.vstack(specs)

edm_flat_specs = get_flat_specs_from(edm_samples, 'song-embeddings-edm')
rock_flat_specs = get_flat_specs_from(rock_samples, 'song-embeddings-rock')
orch_flat_specs = get_flat_specs_from(orch_samples, 'song-embeddings-orchestral')

print('Generated flattened histograms...')

flat_specs = np.vstack((edm_flat_specs, rock_flat_specs, orch_flat_specs))
genre_labels = np.hstack((np.zeros(edm_flat_specs.shape[0]), np.zeros(rock_flat_specs.shape[0]) + 1, np.zeros(orch_flat_specs.shape[0]) + 2))
np.savez_compressed(open_file_in_bucket('raw_flat_specs.npz', 'song-embeddings-genre-classification'), spectrograms=flat_specs, genres=genre_labels)
