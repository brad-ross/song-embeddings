import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from random import shuffle
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

GENRE_ORDER =  [
    'classical',
    'jazz',
    'latin',
    'folk',
    
    'country',
    'funk',
    'indie rock',
    'rock',
    'metal',   
 
    'pop', 
    'r&b',
    'hip hop',
    'rap',
 
    'house',
    'edm',
]

def plot_embedding(embed, labels, plot_type, title="", tsne_params={}, save_path=None, legend=True, label_dict=None):
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
    if N > 10:
        colors = cm.rainbow(np.linspace(0, 1, N))
    scaled_embed = scale(embed)
    
    if plot_type == 'pca':
        pca = PCA(n_components=2)
        pca.fit(scaled_embed)
        #note: will take a while if emebdding is large
        comp1, comp2 = pca.components_
        comp1, comp2 = embed.dot(comp1), embed.dot(comp2)    

    if plot_type == 't-sne':
        tsne = TSNE(**tsne_params)
        comp1, comp2 = tsne.fit_transform(scaled_embed).T

    genres = list(set(labels))
    genres = sorted(genres, key=lambda g: GENRE_ORDER.index(label_dict[g]))
    #genre->indices of that genre (so for loop will change colors)
    g_dict = {i:np.array([j for j in range(len(labels)) if labels[j] == i]) for i in genres}
    for i in range(N):
        g = genres[i]
        if N > 10:
            color = colors[i]
        else:
            color = None
            
        #just use the labels of g as the labels
        plt.scatter(comp1[g_dict[g]], comp2[g_dict[g]],
                    color=color, label='{i}'.format(i=label_dict[g]), alpha=0.7)

    plt.title(title)
    if legend:
        if N < 10:
            lgd = plt.legend(loc='best')
        else:
            lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    if save_path != None:
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
#     plt.show()
