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
import csv

matplotlib.use('TkAgg')


def main():
    times_data = load_data('detection_flags_mrr_bin5_m12_cl61_bin6_m6.h5')  # hdf5 file containing mrr and cl flags
    timelinebar()
    # histogram()


def load_data(fpath):
    with h5py.File(fpath, 'r') as f:
        times = f['time'][:]
        mrr_flags = f['mrr_flag'][:]
        cl_flags = f['cl61_flag'][:]

    # replace -1 values with None because they are considered 'true' in logical and
    mrr_flags = np.where(mrr_flags == np.int64(-1), None, mrr_flags)
    cl_flags = np.where(cl_flags == np.int64(-1), None, cl_flags)
    flag_indices = np.where(np.logical_or(mrr_flags, cl_flags))
    times_flags = times[flag_indices]

    print(times_flags[0], times_flags[-1])

    return times_flags


def gen_synth_data():
    """
    authored by Norm Wood
    """
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
    """
    Kernel Density Estimation Clustering on synthetic data.
    Produced by summing gaussian curves at each bin (more than one data point in a bin means bins are stacked),
    applying bandwidth (larger bandwidth = smoother distribution), clusters defined by local minima and maxima of curve.

    authored by Norm Wood
    """
    # Get the synthetic data
    synth_data = gen_synth_data()
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
    """
    K Means Clustering on synthetic data
    Clustering in k means requires defining the number of clusters ahead of time. This can be done using the
    elbow curve method, where the proper number of clusters is defined at the 'elbow,' or the steepest point on the
    curve.
    K Means did not produce logical clusters on this dataset.
    """
    data = gen_synth_data()
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

    data = {'x': data, 'y': np.ones(data.shape[0]) * 5}
    df = pd.DataFrame(data)
    p = sns.scatterplot(data=df, x="x", y="y", hue=labels, legend="full", palette="bright")
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.1, 1.2), title='Clusters')
    plt.scatter(centroids, np.ones(centroids.shape[0]) * 5, s=50, cmap=['black'])
    print(centroids)
    plt.show()


def write_csv(data, fpath_out='test.csv'):
    with open(fpath_out, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'start_time', 'end_time', 'duration_hr', 'gap_hr'])
        writer.writerows(data)


def timelinebar():
    """Create timeline bar graph using matplotlib broken_barh."""
    fig, ax = plt.subplots()

    rows = []
    with open('snow_events_400_3.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    xranges = []
    for row in rows:
        xranges.append((pd.to_datetime(row[1]), pd.to_datetime(row[2]) - pd.to_datetime(row[1])))

    ax.broken_barh(xranges, [3, 3], facecolor='blue')

    rows = []
    with open('aggregated_snow_events_400_3.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    xranges = []
    for row in rows:
        xranges.append((pd.to_datetime(row[1]), pd.to_datetime(row[2]) - pd.to_datetime(row[1])))
    ax.broken_barh(xranges, [6, 3], facecolor='orange')


    rows = []
    with open('invalid_data_csv/REWRITTEN_REDONE_snow_events_400_3.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    xranges = []
    for row in rows:
        xranges.append((pd.to_datetime(row[1]), pd.to_datetime(row[2]) - pd.to_datetime(row[1])))
    ax.broken_barh(xranges, [9, 3], facecolor='teal')


    rows = []
    with open('aggregated_snow_events_800_2.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    xranges = []
    for row in rows:
        xranges.append((pd.to_datetime(row[1]), pd.to_datetime(row[2]) - pd.to_datetime(row[1])))

    ax.broken_barh(xranges, [12, 3], facecolor='red')

    rows = []
    with open('snow_events_1200_3.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    xranges = []
    for row in rows:
        xranges.append((pd.to_datetime(row[1]), pd.to_datetime(row[2]) - pd.to_datetime(row[1])))

    ax.broken_barh(xranges, [15, 3], facecolor='purple')

    ax.set_yticklabels(['new_data_400', 'aggregated_400', 'redone', 'agg_800', '1200_3'])
    ax.set_yticks([4, 7, 10, 13, 16])
    ax.set_ylim(0, 24)

    plt.show()


def histogram():
    """
    histograms of the durations of the events before and after clustering
    check and see how the distribution of durations changed.
    bin data at 15-minute intervals (0.25 hours)
    :return:
    """
    rows = []
    with open('/Users/sophiareiner/Desktop/copy_redone_snow_events_400_3.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    duration_h = [float(row[3]) for row in rows]
    print(len(duration_h))
    fig, ax = plt.subplots(2, 1)
    fig.tight_layout

    bins = np.arange(0, 60, 0.25)
    ax[0].hist(duration_h, bins, color='teal')
    ax[0].set_title('re clustered data')
    ax[0].set_ylim([0, 450])

    rows = []
    with open('dbscan_csv/snow_events_400_3.csv') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)

    duration_h = [float(row[3]) for row in rows]
    print(len(duration_h))
    ax[1].hist(duration_h, bins, color='blue')
    ax[1].set_title('orig data')
    ax[1].set_ylim([0, 450])

    plt.show()


if __name__ == '__main__':
    main()
