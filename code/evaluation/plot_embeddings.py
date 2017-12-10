import numpy as np
from sklearn.decomposition import PCA
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
    N = len(set(labels))
    pca = PCA(n_components=2)
    pca.fit(embed)
    #note: will take a while if emebdding is large
    comp1, comp2 = pca.components_

    genres = set(labels)
    #genre->indices of that genre (so for loop will change colors)
    g_dict = {i:np.array([j for j in range(len(labels)) if labels[j] == i]) for i in genres}
    for g in genres:
        if label_dict == None:
            #just use the labels of g as the labels
            plt.scatter(embed[g_dict[g]].dot(comp1), embed[g_dict[g]].dot(comp2), \
                       label='{i}'.format(i=g))
        else:
            #use the label_dict labels
            plt.scatter(embed[g_dict[g]].dot(comp1), embed[g_dict[g]].dot(comp2), \
                       label='{i}'.format(i=label_dict[g]))

    plt.title(title)
    if legend:
        plt.legend(loc='best')
    if save_path != None:
        plt.save_fig(save_path)
#     plt.show()
