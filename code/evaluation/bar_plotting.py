import numpy as np
import matplotlib.pyplot as plt


def bar_plot(dict1, dict2, names, title, path):
    plt.figure()
#     categories = dict1.keys()
    categories = [
        'k-means:\nV-measure',
        'k-means:\nAdjusted Mutual Information',
        'Mixture of Gaussians:\nV-measure',
        'Mixture of Gaussians:\nAdjusted Mutual Information',
    ]
    print categories
    N = len(categories)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    scores1 = [dict1[c] for c in categories]
    scores2 = [dict2[c] for c in categories]

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, scores1, width, color='r')
    rects2 = ax.bar(ind + width, scores2, width, color='g')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(categories)
    ax.set_ylim([0, 1])
    ax.legend((rects1[0], rects2[0]), names)
    plt.title(title)
    plt.savefig(path)
    plt.show()

def merge_results(results, metrics):
    def fix_task(m):
        return ' '.join(m.split(' ')[:-1])
    return {fix_task(task) + ':\n' + metric:results[task][metric] for metric in metrics for task in results}



def make_bar_plot(results1, results2, metrics, embed_names, path):
    m_results1 = merge_results(results1, metrics)
    print m_results1
    m_results2 = merge_results(results2, metrics)
    bar_plot(m_results1, m_results2, embed_names, 'Performance on k-means and Mixture of Gaussians', path + '.png')
