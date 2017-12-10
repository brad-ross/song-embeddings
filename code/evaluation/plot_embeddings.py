import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from random import shuffle
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_embedding(embed, labels, title="", save_path=None, legend=True, label_dict=None):
    """
    Projects embedding onto two dimensions, colors according to given label
    @param embed:      embedding matrix
    @param labels:     array of labels for the rows of embed
    @param title:      title of plot
    @param save_path:  path of where to save
    @param legend:     bool to show legend
    @param label_dict: dict that maps labels to real names (eg. {0:'rock', 1:'edm'})


    """
    plt.figure()
    N = len(set(labels))
    scaled_embed = scale(embed)
    #pca = PCA(n_components=2)
    #pca.fit(scaled_embed)
    #note: will take a while if emebdding is large
    #comp1, comp2 = pca.components_
    
    tsne = TSNE(learning_rate=15)
    comp1, comp2 = tsne.fit_transform(scaled_embed).T

    genres = set(labels)
    #genre->indices of that genre (so for loop will change colors)
    g_dict = {i:np.array([j for j in range(len(labels)) if labels[j] == i]) for i in genres}
    g_list = list(genres)
    shuffle(g_list)
    for g in g_list:
        if label_dict == None:
            #just use the labels of g as the labels
            plt.scatter(comp1[g_dict[g]], comp2[g_dict[g]],#embed[g_dict[g]].dot(comp1), embed[g_dict[g]].dot(comp2), \
                       label='{i}'.format(i=g))
        else:
            #use the label_dict labels
            plt.scatter(comp1[g_dict[g]], comp2[g_dict[g]],#embed[g_dict[g]].dot(comp1), embed[g_dict[g]].dot(comp2), \
                       label='{i}'.format(i=label_dict[g]))

    plt.title(title)
    if legend:
        plt.legend(loc='best')
    if save_path != None:
        plt.savefig(save_path)
#     plt.show()
