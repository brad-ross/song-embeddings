import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from random import shuffle
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_embedding(embed, labels, plot_type='t-sne', title="", tsne_params={}, save_path=None, 
                   legend=True, label_dict=None, label_order=None, legend_outside=False, alpha=0.7):
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

    unique_labels = list(set(labels))

    if label_order is not None:
        unique_labels = sorted(unique_labels, key=lambda l: label_order.index(label_dict[l]))
    #genre->indices of that genre (so for loop will change colors)
    l_dict = {i:np.array([j for j in range(len(labels)) if labels[j] == i]) for i in unique_labels}
    for i in range(N):
        l = unique_labels[i]
        color = colors[i]

        #just use the labels of g as the labels
        plt.scatter(comp1[l_dict[l]], comp2[l_dict[l]],
                    color=color, label=label_dict[l], alpha=alpha)

    plt.title(title)
    if legend:
        if N >= 10 or legend_outside:
            lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        else:
            lgd = plt.legend(loc='best')
    if save_path != None:
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

