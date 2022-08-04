import csv
from collections import Counter
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from event_detection import load_data, write_csv
from sklearn import metrics
from IPython.display import display


def main():
    time_flags = load_data('/Users/sophiareiner/PycharmProjects/blizex_tools/blizex_tools/example_analyses/event_detection/event_data/detection_flags_mrr_bin5_cl61_bin6_pip_2022-07-26_REDONE.h5')
    print('time arr len: ', time_flags.shape)
    dbscan_clustering(time_flags)
    # compute_elbow(time_flags)
    # iter_dbscan(time_flags)


def iter_dbscan(data):
    """
    loop through different values of epsilon and min_samples to determine the optimal values
    For this dataset, looking for clusters of dense data and outliers where data is less dense.
    note: this takes a while if you use a lot of data.
    """
    X = data.reshape(-1, 1)
    eps_vals = np.arange(1500, 3500, 500)
    min_samples = np.array([3])
    dbscan_params = list(product(eps_vals, min_samples))

    no_of_clusters = []
    sil_score = []
    epsvalues = []
    min_samp = []

    for p in dbscan_params:
        cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(X)
        epsvalues.append(p[0])
        min_samp.append(p[1])
        no_of_clusters.append(len(np.unique(cluster.labels_)))
        sil_score.append(metrics.silhouette_score(X, cluster.labels_))

    pca_eps_min = list(zip(no_of_clusters, sil_score, epsvalues, min_samp))
    pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=[
        'no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])

    display(pca_eps_min_df)


def compute_elbow(data):
    """
    Using an elbow curve is another way to determine the optimal epsilon.
    compute epsilon: look at the 'knee' of the graph to find the best epsilon point. The elbow curve indicated
    an optimal eps value of about 400.
    """
    X = data.reshape(-1, 1)

    # n_neighbors = 5 as neighbors function returns distance of point to itself (i.e. first column will be zeros)
    nbrs = NearestNeighbors(n_neighbors=4).fit(X)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(X)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sort_neigh_dist[:, 3]
    plt.plot(k_dist)
    plt.axhline(y=400, linewidth=1, linestyle='dashed', color='k')
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (2th NN)")
    plt.show()


# https://www.reneshbedre.com/blog/dbscan-python.html
def dbscan_clustering(data):
    """
    DBScan clustering:
    dbscan clusters on the basis that clusters tend to be in denser areas
    and outliers tend to be in non densely populated areas. The user must define
    an epsilon value, which is the radius around a core point that the algorithm searches
    for other data points. The user must define min_samples, which is defined to be
    the minimum number of data points within the epsilon circle to be considered a cluster.
    Note that the core point is included in the number of required min_samples.
    """
    # data = DATA
    # data = gen_synth_data()
    X = data.reshape(-1, 1)

    dbscan = DBSCAN(eps=1200, min_samples=2)
    model = dbscan.fit(X)
    labels = model.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(type(labels))
    print(f'estimated number of clusters: {n_clusters_}')
    print(f'estimated number of noise points: {n_noise_}')
    unique_labels = set(labels)
    print(f'unique labels: {unique_labels}')
    print(f'Points per cluster: {Counter(labels)}')

    # turn unix to datetime
    data = data * np.timedelta64(1, 's') + np.datetime64('1970-01-01T00:00:00Z')

    # create csv of detected events
    eval_events(labels, n_clusters_, data, 'snow_events_1200_2.csv')

    # data = {'x': data, 'y': np.ones(data.shape[0]) * 5}
    # df = pd.DataFrame(data)

    # p = sns.scatterplot(data=df, x="x", y="y", hue=labels, legend="full", palette="bright")
    # sns.move_legend(p, "upper right", bbox_to_anchor=(1.1, 1.2), title='Clusters')
    # plt.show()
    # print(metrics.silhouette_score(X, labels))  # takes a long time for a lot of data.


def eval_events(labels: np.ndarray, num_clusters: int, times: np.ndarray, f_out='test.csv'):
    """
    Write clusters created by dbscan into a csv file. csv includes start/end of cluster,
    duration of cluster, and time gap between clusters
    :param labels: id of each cluster, -1 indicates outliers
    :param num_clusters: number of clusters
    :param times: 1D time data
    :param f_out: name of new file
    """
    rows = []
    events = []
    for i in range(num_clusters - 1):
        indices = np.where(labels == i)  # np.where returns a tuple
        events.append((indices[0][0], indices[0][-1]))

    for i in range(len(events)):
        t_start = times[events[i][0]]
        t_end = times[events[i][-1]]
        try:
            t_next = times[events[i + 1][0]]
            gap = np.timedelta64(t_next - t_end, 's')
            gap_h = np.int32(gap) / 3600
        except IndexError:
            gap_h = -999

        duration = np.timedelta64(t_end - t_start)
        duration_h = np.int32(duration) / 3600

        row = f'{i} {t_start} {t_end} {duration_h:.3f} {gap_h:.3f}'
        rows.append(row.split())
    write_csv(rows, f_out)
    print('done')


if __name__ == '__main__':
    main()
