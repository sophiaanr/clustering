import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from event_detection import load_data
import numpy as np
from event_detection import write_csv


def main():
    time_flags = load_data('detection_flags_mrr_bin5_m12_cl61_bin6_m6.h5')
    kde_clustering(time_flags)
    # estimate_bandwidth(time_flags)


def estimate_bandwidth(data):
    X = data.reshape(-1, 1)
    bandwidth = np.arange(3300, 3400, 20)
    kde = KernelDensity(kernel='gaussian')
    grid = GridSearchCV(kde, {'bandwidth': bandwidth})
    grid.fit(X)
    kde = grid.best_estimator_
    print(kde.bandwidth)


# find better way to convert unixtime to datetime
def kde_clustering(data):
    X = data.reshape(-1, 1)
    plt.scatter(data, np.ones(data.shape[0]) * -13.5, s=1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3385.5)  # bandwidth calculated from all the data I think??
    kde.fit(X)

    # arrange times from start time to end time, with gap of 4hours
    s = np.arange(1639416510, 1652170150 + 5400, 5400)  # what does this do????  1640251340 1643243120
    e = kde.score_samples(s.reshape(-1, 1))

    plt.plot(s, e)
    plt.show()

    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

    # labels = []  # create labels like in dbscan
    events = []
    cluster = (X[X < s[mi][0]]) * np.timedelta64(1, 's') + np.datetime64('1970-01-01T00:00:00Z')  # first cluster
    events.append((cluster[0], cluster[-1]))
    # labels.append(list(np.zeros(len(cluster))))
    # print(f'{cluster[0]} {cluster[-1]} {np.int64(np.timedelta64(cluster[-1]-cluster[0], "s")) / 3600:.3f}')
    for i in range(len(mi) - 1):  # middle clusters
        cluster = (X[(X >= s[mi][i]) * (X <= s[mi][i + 1])]) * np.timedelta64(1, 's') + np.datetime64(
            '1970-01-01T00:00:00Z')
        events.append((cluster[0], cluster[-1]))
        # labels.append(list(np.ones(len(cluster)) * (i+1)))
        # print(f'{cluster[0]} {cluster[-1]} {np.int64(np.timedelta64(cluster[-1] - cluster[0], "s")) / 3600:.3f}')
    cluster = (X[X >= s[mi][-1]]) * np.timedelta64(1, 's') + np.datetime64('1970-01-01T00:00:00Z')  # end cluster
    events.append((cluster[0], cluster[-1]))
    # labels.append(list(np.ones(len(cluster)) * (len(mi))))
    # print(f'{cluster[0]} {cluster[-1]} {np.int64(np.timedelta64(cluster[-1]-cluster[0], "s")) / 3600:.3f}')


    rows = []
    for i in range(len(events)):
        t_start = events[i][0]
        t_end = events[i][1]

        try:
            t_next = events[i + 1][0]
            gap = np.timedelta64(t_next - t_end, 's')
            gap_h = np.int32(gap) / 3600
        except IndexError:
            gap_h = -999

        duration = np.timedelta64(t_end - t_start)
        duration_h = np.int32(duration) / 3600

        row = f'{i} {t_start} {t_end} {duration_h:.3f} {gap_h:.3f}'
        rows.append(row.split())
    write_csv(rows, 'event_kde_bdw3385.5_2.csv')

    # labels = [int(x) for xs in labels for x in xs]  # convert list of lists to flat list
    data = {'x': data[:10000] * np.timedelta64(1, 's') + np.datetime64('1970-01-01T00:00:00Z'),
            'y': (np.ones(data.shape[0]) * 5)[:10000]}
    df = pd.DataFrame(data)

    # something wrong here... the label length doesn't match the data length
    p = sns.scatterplot(data=df, x="x", y="y", hue=labels[:10000], legend="full", palette="bright")
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.1, 1.2), title='Clusters')
    plt.show()


if __name__ == '__main__':
    main()
