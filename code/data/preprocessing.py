from ..data.cloud_storage import get_path_to_bucket, list_files_in_bucket, open_file_in_bucket
from ..data.audio import mp3_to_array, get_spectrogram
from random import shuffle, randrange
import os
from multiprocessing import Pool, Process
import numpy as np
from ..utils import log

DATA_BATCH_SIZE = 12000

def get_nonempty_files(files, bucket):
    nonempty = []
    for f in files:
	if os.path.getsize(os.path.join(get_path_to_bucket(bucket), f)) == 0:
	    continue
	
        nonempty.append(f)
    
    return nonempty

log('retrieving raw previews...')


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

def create_data_batch(previews_to_process, input_bucket, output_bucket, filename):
    specs = get_rand_specs_from(previews_to_process, input_bucket, 5)
    log('{}, {}'.format(specs.shape, specs.nbytes))
    np.save(open_file_in_bucket('{}.npy'.format(filename), output_bucket), specs)
    del specs

def generate_dataset(name, previews_prefix, previews_bucket, output_bucket, num_samples_per_song, shuffle=True):
    all_preview_files = [f for f in list_files_in_bucket(previews_bucket) if f != 'dirEmptyCheck' and f.startswith(previews_prefix)]
    log('{} total preview files'.format(len(all_preview_files)))
    raw_preview_files = all_preview_files * num_samples_per_song 
    if shuffle:
        shuffle(raw_preview_files)

    with open_file_in_bucket('{name}_labels.csv'.format(name=name), output_bucket) as labels_file:
        for p in raw_preview_files:
            line = p.split('.mp3')[0].replace(',', ';').replace('_', ',')
            labels_file.write(line + '\n')

    log('{} preview samples'.format(len(raw_preview_files)))

    num_batches = len(raw_preview_files)/DATA_BATCH_SIZE + 1
    for start, end in [(i*DATA_BATCH_SIZE, min((i+1)*DATA_BATCH_SIZE, len(raw_preview_files))) for i in range(num_batches)]:
        p = Process(target=create_data_batch, args=(raw_preview_files[start:end], previews_bucket, output_bucket, '{}_{}_{}'.format(name, start, end)))
        p.start()
        p.join()

generate_dataset('billyjoel', 'Billy Joel', 'song-embeddings-artist-experiments-previews', 'song-embeddings-artist-experiments', 10, shuffle=False)

