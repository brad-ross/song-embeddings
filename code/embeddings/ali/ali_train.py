from ali_model import ALIModel
from keras_adversarial import gan_targets
import numpy as np

model = ALIModel().model
data = np.load('./test_0_1000.npy')

batch_size = 101

def get_batch_range(batch_num, batch_size, data_size):
    return (batch_num*batch_size, min((batch_num+1)*batch_size, data_size))

num_batches = data.shape[0]/batch_size + 1
batch_indices = [
    get_batch_range(i, batch_size, data.shape[0]) for i in range(num_batches)
]

for start, end in batch_indices:
    print model.train_on_batch(data[start:end, :, :, None], gan_targets(end - start))
