import numpy as np
from ali_model_l2 import ALIModel
from ...data.cloud_storage import open_file_in_bucket, get_path_to_file_in_bucket
#from ali_train_l2 import rescale_batch

def load_model_from(model_class, weights_file):
    model = model_class()
    model.load_weights_from_file(get_path_to_file_in_bucket(weights_file, 'song-embeddings-dataset'))
    return model

def create_embedding_fn(model_class, weights_file):
    model = load_model_from(model_class, weights_file)
    
    def generate_embedding(specs):
        specs_with_depth = specs[:, :, :, None]
        return np.reshape(model.encoder.predict(specs_with_depth), (specs.shape[0], -1))

    return generate_embedding
