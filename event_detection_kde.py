#!/usr/bin/env python3
# See for example
# https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit/35151947#35151947
# and
# https://stackoverflow.com/questions/60355497/choosing-bandwidthlinspace-for-kernel-density-estimation-why-my-bandwidth-doe


import random
import seaborn as sns
import numpy as np
import sklearn.neighbors
import sklearn.model_selection
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KernelDensity
import pandas as pd
from collections import Counter
import h5py

matplotlib.use('TkAgg')

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


def load_data(fpath):
    with h5py.File(fpath, 'r') as f:
        times = f['time'][100000:400000]
        mrr_flags = f['mrr_flag'][100000:400000]
        cl_flags = f['cl61_flag'][100000:400000]

    # convert unixtime to datetime - it may just be more beneficial to cluster in unixtime rather than datetime.
    # times = times * np.timedelta64(1, 's') + np.datetime64('1970-01-01T00:00:00Z')

    # replace -1 values with 0 because they are considered 'true' in logical and
    mrr_flags = np.where(mrr_flags == np.int64(-1), None, mrr_flags)
    cl_flags = np.where(cl_flags == np.int64(-1), None, cl_flags)
    flag_indices = np.where(np.logical_or(mrr_flags, cl_flags))
    times_flags = times[flag_indices]

    print(times_flags[0], times_flags[-1])

    return times_flags


def gen_synth_data():
    time_list = []
    N_gaussians = 20
    N_samples = 50
    centers = random.sample(range(10000), N_gaussians)
    widths = random.sample(range(200), N_gaussians)
    for i_gauss in range(N_gaussians):
        for i_sample in range(N_samples):
            randomnumber = round(random.gauss(centers[i_gauss], widths[i_gauss]))
            if randomnumber not in time_list:
                time_list.append(randomnumber)
    time_list.sort()
    times = np.array(time_list)
    return times


def KDE_clustering():
    # Get the synthetic data
    synth_data = DATA
    # Show the synthetic data
    plt.scatter(synth_data, np.ones(synth_data.shape[0]) * 5, s=5)
    # plt.show(block=False)
    # matplotlib.pyplot.clf()

    # print(synth_data)

    # Do the kernel density estimation
    # params = {"bandwidth": numpy.logspace(-2, 0.5, 20)}
    # grid = sklearn.model_selection.GridSearchCV(sklearn.neighbors.KernelDensity(), params)
    # grid.fit(synth_data.reshape(-1,1))

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    kde = sklearn.neighbors.KernelDensity(kernel="gaussian", bandwidth=3).fit(synth_data.reshape(-1, 1))
    # kde = grid.best_estimator_

    s = np.linspace(0, 5000)
    e = kde.score_samples(s.reshape(-1, 1))
    idx_min = scipy.signal.argrelextrema(e, np.less)[0]
    idx_max = scipy.signal.argrelextrema(e, np.greater)[0]

    print(idx_min)
    print(idx_max)
    print("Minima:  ", s[idx_min])
    print("Maxima:  ", s[idx_max])

    plt.plot(s, e)
    plt.show()


def KMeans_clustering():
    data = DATA
    X = data.reshape(-1, 1)

    sum_of_squared_distances = []
    K = range(1, 100)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    kmeans = KMeans(n_clusters=6, n_init=10)
    a = kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    colors = ['g.', 'r.', 'y.', 'b.']
    # for i in range(len(X)):
    #     plt.plot(X[i], colors[labels[i]], markersize=5)

    data = {'x': data, 'y': np.ones(data.shape[0]) * 5}
    df = pd.DataFrame(data)
    p = sns.scatterplot(data=df, x="x", y="y", hue=labels, legend="full", palette="bright")
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.1, 1.2), title='Clusters')
    plt.scatter(centroids, np.ones(centroids.shape[0]) * 5, s=50, cmap=['black'])
    print(centroids)
    plt.show()





if __name__ == '__main__':
    times_data = load_data('detection_flags_mrr_bin5_m12_cl61_bin6_m6.h5')
    KDE_clustering()
    # KMeans_clustering()
