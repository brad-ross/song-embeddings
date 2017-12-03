from ..data.cloud_storage import get_path_to_bucket, list_files_in_bucket, open_file_in_bucket
from ..data.audio import mp3_to_array, get_spectrogram
from random import shuffle, randrange
import os
from multiprocessing import Pool, Process
import numpy as np
from ..utils import log

NUM_SAMPLES_PER_SONG = 10
DATA_BATCH_SIZE = 2000

def get_nonempty_files(files, bucket):
    nonempty = []
    for f in files:
	if os.path.getsize(os.path.join(get_path_to_bucket(bucket), f)) == 0:
	    continue
	
        nonempty.append(f)
    
    return nonempty

log('retrieving raw previews...')

all_preview_files = list_files_in_bucket('song-embeddings-raw-previews')
raw_preview_files = get_nonempty_files(all_preview_files, 'song-embeddings-raw-previews') * NUM_SAMPLES_PER_SONG
shuffle(raw_preview_files)

labels_file = open_file_in_bucket('labels.csv', 'song-embeddings-dataset')
for p in raw_preview_files:
    genre, id = p.split('.')[0].split('_')
    labels_file.write('{id},{genre}\n'.format(id=id, genre=genre))
labels_file.close()

log('{} preview samples'.format(len(raw_preview_files)))

def get_flat_rand_seq(args):
    f, bucket, num_secs = args
    with open_file_in_bucket(f, bucket) as mp3:
        mp3_array, rate = mp3_to_array(mp3)
        seg_size = rate * num_secs
        rand_start = randrange(mp3_array.size - 1 - seg_size)
        freqs, times, spec = get_spectrogram(mp3_array[rand_start:(rand_start + seg_size)], rate)
        return spec

def get_rand_specs_from(files, bucket, num_secs): 
    arg_tups = zip(*[files, [bucket] * len(files), [num_secs] * len(files)])
    pool = Pool(processes=8)
    specs = pool.map(get_flat_rand_seq, arg_tups)
    stacked = np.array(specs)
    return stacked

def create_data_batch(previews_to_process, start, end):
    specs = get_rand_specs_from(previews_to_process, 'song-embeddings-raw-previews', 5)
    log('{}, {}'.format(specs.shape, specs.nbytes))
    np.save(open_file_in_bucket('{}_{}.npy'.format(start, end), 'song-embeddings-dataset'), specs)
    del specs

def generate_dataset():
    num_batches = len(raw_preview_files)/DATA_BATCH_SIZE + 1
    for start, end in [(i*DATA_BATCH_SIZE, min((i+1)*DATA_BATCH_SIZE, len(raw_preview_files))) for i in range(num_batches)]:
        p = Process(target=create_data_batch, args=(raw_preview_files[start:end], start, end))
        p.start()
        p.join()

generate_dataset()

log('Generated flattened histograms...')
