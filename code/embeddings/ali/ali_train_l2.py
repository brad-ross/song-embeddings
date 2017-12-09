from ali_model_l2 import ALIModel
from keras_adversarial import gan_targets
import numpy as np
from ...data.cloud_storage import open_file_in_bucket, get_path_to_file_in_bucket

def get_batch_range(batch_num, batch_size, data_size):
    return (batch_num*batch_size, min((batch_num+1)*batch_size, data_size))

def rescale_batch(batch):
    batch += -(np.min(batch))
    batch /= np.max(batch / (1 - (-1))
    batch += (-1)
    return batch

def train_on_data_batch(model, data_file, print_interval):
    data = np.load(open_file_in_bucket(data_file, 'song-embeddings-dataset'))
    num_batches = data.shape[0]/batch_size + 1
    batch_indices = [
        get_batch_range(i, batch_size, data.shape[0]) for i in range(num_batches)
    ]
    for i in range(len(batch_indices)):
        start, end = batch_indices[i]
	#batch = rescale_batch(data[start:end, :, :, None])
	batch = data[start:end, :, :, None]
	targets = gan_targets(end - start)
	targets[0] *= np.random.uniform(0.7, 0.9, end - start)[:, None]
	targets[3] *= np.random.uniform(0.7, 0.9, end - start)[:, None]
        losses = model.train_on_batch(batch, targets)
	if i % print_interval == 0:
	    print losses
	    print np.mean(np.reshape(model.predict(data[start:end, :, :, None]), (4, -1)), axis=1)

def train_one_epoch(model, data_files, print_interval):
    for f in data_files:
        print 'training on {}'.format(f)
        train_on_data_batch(model, f, print_interval)

def save_weights(model, weight_file):
    model.save_weights(get_path_to_file_in_bucket(weight_file, 'song-embeddings-dataset'))

model = ALIModel().model
batch_size = 101
epochs = 5
print_interval = 10

print(model.metrics_names)
for e in range(epochs):
    print 'epoch {}'.format(e)
    train_one_epoch(model, ['0_12000.npy', '12000_24000.npy', '24000_36000.npy', '36000_48000.npy', 
                           '48000_60000.npy', '60000_72000.npy', '72000_84000.npy', '84000_96000.npy',
                           '96000_108000.npy'], print_interval)
    save_weights(model, 'model_weights_7_epoch_{}'.format(e))
