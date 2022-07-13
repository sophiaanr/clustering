from collections import Counter
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from event_detection_kde import load_data
from sklearn import metrics
from IPython.display import display
import csv

DATA = np.array([-52, -47, -23, -16, -1, 4, 6, 15, 16,
                 24, 32, 39, 44, 51, 53, 54, 57, 66,
                 67, 68, 76, 78, 80, 82, 90, 91, 93,
                 102, 108, 109, 115, 119, 123, 126, 140, 145,
                 146, 151, 153, 165, 195, 206, 230, 235, 241,
                 318, 351, 354, 369, 374, 375, 380, 382, 383,
                 391, 395, 400, 403, 404, 405, 411, 413, 418,
                 419, 420, 421, 426, 428, 429, 431, 441, 443,
                 447, 449, 451, 452, 465, 468, 469, 471, 483,
                 491, 495, 498, 499, 514, 516, 1096, 1126, 1179,
                 1191, 1266, 1361, 1379, 1393, 1405, 1407, 1419, 1427,
                 1430, 1457, 1470, 1474, 1478, 1486, 1500, 1513, 1519,
                 1520, 1525, 1529, 1535, 1565, 1570, 1573, 1575, 1579,
                 1584, 1585, 1586, 1598, 1603, 1613, 1631, 1639, 1641,
                 1658, 1659, 1665, 1674, 1695, 1696, 1705, 1707, 1722,
                 1727, 1733, 1735, 1737, 1739, 1747, 1753, 1764, 1767,
                 1774, 1777, 1780, 1785, 1787, 1790, 1801, 1802, 1804,
                 1806, 1808, 1812, 1819, 1821, 1824, 1826, 1831, 1832,
                 1837, 1842, 1843, 1844, 1853, 1854, 1856, 1862, 1866,
                 1875, 1882, 1884, 1888, 1907, 1915, 1959, 3252, 3281,
                 3288, 3289, 3295, 3302, 3309, 3312, 3313, 3316, 3318,
                 3322, 3324, 3327, 3328, 3329, 3336, 3341, 3342, 3345,
                 3346, 3347, 3350, 3352, 3353, 3354, 3358, 3359, 3361,
                 3367, 3368, 3371, 3375, 3378, 3379, 3381, 3384, 3385,
                 3387, 3389, 3396, 3405, 3444, 3541, 3543, 3544, 3576,
                 3582, 3585, 3598, 3604, 3606, 3612, 3623, 3634, 3664,
                 3680, 3688, 3689, 3695, 3696, 3701, 3703, 3704, 3718,
                 3722, 3731, 3744, 3754, 3755, 3756, 3758, 3761, 3772,
                 3785, 3786, 3799, 3801, 3803, 3816, 3817, 3821, 3823,
                 3826, 3833, 3842, 3845, 3846, 3847, 3849, 3850, 3851,
                 3856, 3864, 3865, 3873, 3883, 3887, 3896, 3897, 3901,
                 3903, 3906, 3909, 3910, 3911, 3918, 3920, 3922, 3924,
                 3925, 3928, 3931, 3936, 3939, 3954, 3956, 3960, 3962,
                 3966, 3976, 3977, 3991, 3994, 4016, 4017, 4037, 4041,
                 4042, 4044, 4053, 4109, 4115, 4130, 4162, 4886, 4972,
                 5027, 5054, 5060, 5067, 5125, 5138, 5141, 5165, 5186,
                 5196, 5251, 5264, 5278, 5293, 5296, 5307, 5309, 5313,
                 5329, 5338, 5349, 5366, 5368, 5400, 5403, 5413, 5414,
                 5416, 5447, 5452, 5457, 5461, 5469, 5481, 5490, 5498,
                 5549, 5555, 5557, 5564, 5571, 5584, 5592, 5604, 5668,
                 5671, 5684, 5688, 5696, 5700, 5721, 5726, 5731, 5736,
                 5739, 5751, 5772, 5775, 5776, 5814, 5823, 5824, 5828,
                 5829, 5836, 5851, 5855, 5859, 5869, 5871, 5883, 5891,
                 5895, 5900, 5901, 5902, 5903, 5909, 5913, 5914, 5922,
                 5944, 5945, 5971, 5981, 5989, 5997, 6003, 6012, 6024,
                 6044, 6074, 6111, 6478, 6498, 6557, 6584, 6585, 6595,
                 6617, 6622, 6625, 6634, 6635, 6644, 6651, 6655, 6656,
                 6657, 6658, 6661, 6662, 6667, 6673, 6674, 6675, 6680,
                 6683, 6686, 6690, 6696, 6697, 6699, 6701, 6704, 6706,
                 6707, 6711, 6722, 6728, 6730, 6731, 6734, 6736, 6737,
                 6738, 6741, 6742, 6743, 6744, 6753, 6754, 6755, 6761,
                 6763, 6765, 6768, 6776, 6779, 6782, 6783, 6786, 6796,
                 6801, 6804, 6806, 6807, 6812, 6813, 6816, 6821, 6823,
                 6825, 6827, 6841, 6842, 6843, 6848, 6852, 6857, 6862,
                 6867, 6870, 6892, 6905, 6916, 6992, 7048, 7062, 7064,
                 7097, 7098, 7102, 7104, 7107, 7118, 7119, 7130, 7131,
                 7133, 7139, 7144, 7146, 7147, 7152, 7155, 7159, 7164,
                 7171, 7178, 7179, 7183, 7188, 7193, 7207, 7212, 7218,
                 7219, 7224, 7229, 7244, 7245, 7252, 7254, 7258, 7259,
                 7265, 7266, 7268, 7275, 7291, 7304, 7314, 7320, 7346,
                 7355, 7384, 7393, 7409, 7423, 7424, 7426, 7467, 7470,
                 7476, 7481, 7483, 7485, 7489, 7493, 7498, 7515, 7520,
                 7523, 7528, 7529, 7534, 7535, 7538, 7539, 7540, 7543,
                 7544, 7545, 7551, 7552, 7556, 7557, 7558, 7559, 7561,
                 7562, 7563, 7567, 7569, 7570, 7572, 7573, 7574, 7576,
                 7578, 7580, 7581, 7582, 7584, 7585, 7590, 7593, 7595,
                 7597, 7601, 7602, 7609, 7610, 7612, 7616, 7618, 7619,
                 7620, 7626, 7637, 7648, 7650, 7656, 7658, 7663, 7666,
                 7671, 7689, 7702, 7704, 7706, 7707, 7728, 7747, 7765,
                 7777, 7791, 7803, 7809, 7818, 7823, 7830, 7834, 7841,
                 7848, 7849, 7854, 7863, 7882, 7890, 7907, 7922, 7925,
                 7948, 7950, 7959, 7962, 7963, 7995, 8001, 8035, 8038,
                 8040, 8041, 8046, 8047, 8051, 8082, 8085, 8090, 8113,
                 8130, 8142, 8143, 8168, 8304, 8476, 8491, 8513, 8521,
                 8524, 8527, 8529, 8537, 8542, 8543, 8545, 8555, 8558,
                 8560, 8564, 8565, 8567, 8568, 8572, 8576, 8579, 8580,
                 8582, 8583, 8584, 8587, 8589, 8596, 8597, 8600, 8601,
                 8602, 8609, 8610, 8614, 8615, 8619, 8622, 8633, 8638,
                 8640, 8642, 8653, 8661, 8668, 8678, 8685, 8687, 8696,
                 8701, 8713, 8722, 8730, 8731, 8735, 8738, 8744, 8751,
                 8752, 8755, 8761, 8765, 8768, 8770, 8771, 8773, 8774,
                 8777, 8780, 8783, 8784, 8789, 8798, 8800, 8801, 8802,
                 8813, 8815, 8822, 8835, 8836, 8847, 8860, 9139, 9153,
                 9162, 9172, 9185, 9189, 9195, 9202, 9203, 9206, 9208,
                 9209, 9210, 9213, 9214, 9217, 9218, 9225, 9227, 9237,
                 9239, 9242, 9243, 9246, 9249, 9256, 9257, 9258, 9259,
                 9264, 9265, 9267, 9270, 9272, 9274, 9276, 9278, 9280,
                 9283, 9287, 9288, 9291, 9292, 9298, 9306, 9315, 9316,
                 9337, 9433, 9443, 9445, 9448, 9449, 9454, 9469, 9476,
                 9481, 9486, 9488, 9507, 9510, 9512, 9518, 9520, 9524,
                 9525, 9531, 9534, 9535, 9536, 9539, 9553, 9558, 9565,
                 9566, 9569, 9573, 9585, 9588, 9592, 9596, 9608, 9613,
                 9615, 9617, 9621, 9628, 9634, 9637, 9639, 9640, 9641,
                 9643, 9644, 9672, 9693, 9697, 9698, 9706, 9708, 9717,
                 9743, 9744, 9750, 9754, 9757, 9760, 9769, 9771, 9777,
                 9784, 9788, 9792, 9800, 9804, 9805, 9806, 9809, 9810,
                 9811, 9823, 9827, 9828, 9859, 9863, 9864, 9883, 9890,
                 9891, 9896, 9903, 9904, 9921, 9945, 9946, 9947, 9972,
                 9984, 9990, 10013, 10035, 10095])


def iter_dbscan(data):
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


# https://www.reneshbedre.com/blog/dbscan-python.html
def dbscan_clustering(data):
    # data = DATA
    # data = gen_synth_data()
    X = data.reshape(-1, 1)

    ## doesn't really work for this dataset.
    ## compute epsilon: look at the 'knee' of the graph to find the best epsilon point
    ## n_neighbors = 5 as neighbors function returns distance of point to itself (i.e. first column will be zeros)
    # nbrs = NearestNeighbors(n_neighbors=4).fit(X)
    ## Find the k-neighbors of a point
    # neigh_dist, neigh_ind = nbrs.kneighbors(X)
    ## sort the neighbor distances (lengths to points) in ascending order
    ## axis = 0 represents sort along first axis i.e. sort along row
    # sort_neigh_dist = np.sort(neigh_dist, axis=0)
    # k_dist = sort_neigh_dist[:, 3]
    # plt.plot(k_dist)
    # plt.axhline(y=400, linewidth=1, linestyle='dashed', color='k')
    # plt.ylabel("k-NN distance")
    # plt.xlabel("Sorted observations (2th NN)")
    # plt.show()

    dbscan = DBSCAN(eps=1550, min_samples=3)
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

    # create csv of detected events
    # eval_events(labels, n_clusters_, data)

    data = {'x': data * np.timedelta64(1, 's') + np.datetime64('1970-01-01T00:00:00Z'), 'y': np.ones(data.shape[0]) * 5}
    df = pd.DataFrame(data)

    p = sns.scatterplot(data=df, x="x", y="y", hue=labels, legend="full", palette="bright")
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.1, 1.2), title='Clusters')
    plt.show()
    # print(metrics.silhouette_score(X, labels)) takes a long time for a lot of data.


def write_csv(data, fpath_out='test.csv'):
    with open(fpath_out, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'start_time', 'end_time', 'duration_hr', 'gap_hr'])
        writer.writerows(data)


def eval_events(labels: np.ndarray, num_clusters: int, times: np.ndarray):
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
    write_csv(rows)
    print('done')


def timelinebar():
    fig, ax = plt.subplots()

    rows = []
    with open('test.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    xranges = []
    for row in rows:
        xranges.append((pd.to_datetime(row[1]), pd.to_datetime(row[2]) - pd.to_datetime(row[1])))

    ax.broken_barh(xranges, [9, 3], facecolor='orange')

    ax.set_yticklabels([''])
    ax.set_yticks([10])
    ax.set_ylim(0, 24)

    plt.show()


if __name__ == '__main__':
    time_flags = load_data('detection_flags_mrr_bin5_m12_cl61_bin6_m6.h5')
    dbscan_clustering(time_flags)
    # timelinebar()
    # iter_dbscan(time_flags)
