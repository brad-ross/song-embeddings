import pandas as pd
import numpy as np

def get_numerical_labels(f_path):
    labels = pd.read_csv(f_path, header=None)
    return np.array((labels.iloc[:,1]).astype('category').cat.codes)


