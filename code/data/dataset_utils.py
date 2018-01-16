import pandas as pd
import numpy as np

def get_numerical_labels(f_path, label_col):
    labels = pd.read_csv(f_path, header=None)
    labels_cat = (labels.iloc[:,label_col]).astype('category').cat
    label_mapping = dict(enumerate(labels_cat.categories))
    inv_label_mapping = {a: l for l, a in label_mapping.iteritems()}
    return label_mapping, inv_label_mapping, np.array(labels_cat.codes)

