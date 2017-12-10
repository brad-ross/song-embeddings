import numpy as np
import matplotlib.pyplot as plt


def bar_plot(dict1, dict2, categories, names, title, path):
    plt.figure()
    N = len(categories)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    scores1 = [dict1[c] for c in categories]
    scores2 = [dict2[c] for c in categories]

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, scores1, width, color='r')
    rects2 = ax.bar(ind + width, scores2, width, color='g')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(categories)
    ax.set_ylim([0, 1])
    ax.legend((rects1[0], rects2[0]), names)
    plt.title(title)
    plt.savefig(path)
    # plt.show()

def plot_for_task(results1, results2, metrics, names, path, task):
    dict1 = {m:results1[task][m] for m in results1[task] if m in metrics}
    dict2 = {m:results2[task][m] for m in results2[task] if m in metrics}
    bar_plot(dict1, dict2, metrics, names, 'Performance on ' + task, path)

def make_bar_plot(results1, results2, metrics, embed_names, path):
    plot_for_task(results1, results2, metrics, embed_names, path + '_kmeans.png', 'k-means Task')
    plot_for_task(results1, results2, metrics, embed_names, path + '_mog.png', 'Mixture of Gaussians Task')
