#!/usr/bin/env python3
# See for example
# https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit/35151947#35151947
# and
# https://stackoverflow.com/questions/60355497/choosing-bandwidthlinspace-for-kernel-density-estimation-why-my-bandwidth-doe


import numpy
import random
import sklearn.neighbors
import sklearn.model_selection
import matplotlib
import matplotlib.pyplot
import scipy.signal

matplotlib.use('TkAgg')


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
    times = numpy.array(time_list)
    return times


# Get the synthetic data

synth_data = gen_synth_data()
# Show the synthetic data
matplotlib.pyplot.scatter(synth_data, numpy.ones(synth_data.shape[0]) * 5)
matplotlib.pyplot.show(block=False)
# matplotlib.pyplot.clf()

print(synth_data)

# Do the kernel density estimation
# params = {"bandwidth": numpy.logspace(-2, 0.5, 20)}
# grid = sklearn.model_selection.GridSearchCV(sklearn.neighbors.KernelDensity(), params)
# grid.fit(synth_data.reshape(-1,1))

# print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

kde = sklearn.neighbors.KernelDensity(kernel="gaussian", bandwidth=3.).fit(synth_data.reshape(-1, 1))
# kde = grid.best_estimator_

s = numpy.linspace(0, 10000)
e = kde.score_samples(s.reshape(-1, 1))
idx_min = scipy.signal.argrelextrema(e, numpy.less)[0]
idx_max = scipy.signal.argrelextrema(e, numpy.greater)[0]

print(idx_min)
print(idx_max)
print("Minima:  ", s[idx_min])
print("Maxima:  ", s[idx_max])

matplotlib.pyplot.plot(s, e)
matplotlib.pyplot.show()
