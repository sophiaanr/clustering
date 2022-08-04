#!/usr/bin/env python3

import numpy
from glob import glob
import h5py
import csv


def unix_to_datetime64(unix_time):
    return unix_time * numpy.timedelta64(1, 's') + numpy.datetime64('1970-01-01T00:00:00Z')


with open('aggregated_snow_events_400_3.csv', 'r') as f_in:
    lines = f_in.readlines()

t_start_list = []
t_end_list = []

for line in lines[1:]:
    parts = line.split(',')
    t_start_list.append(numpy.datetime64(parts[1]))
    t_end_list.append(numpy.datetime64(parts[2]))


fpaths = sorted(glob('/Users/sophiareiner/Documents/BlizExData/event_files_h5/*.h5'))
if len(fpaths) != len(lines[1:]):
    raise ValueError('length of events in csv does not match number of event files')

mrr_count = []
cl61_count = []
mrr_AND_cl61 = []
mrr_OR_cl61 = []

for path in fpaths:
    with h5py.File(path, 'r') as f:
        mrr_flags = numpy.array(f['mrr_flag'])
        cl61_flags = numpy.array(f['cl61_flag'])
        time_flag = numpy.array(unix_to_datetime64(f['time']))

    if len(mrr_flags) != len(time_flag) or len(cl61_flags) != len(time_flag):
        raise ValueError('length of flags from hdf5 file are not the same length')

    obs_count = len(mrr_flags)

    # count occurrences of 1 in flags
    mrr_ct = numpy.count_nonzero(mrr_flags == 1)
    mrr_count.append(mrr_ct / obs_count)

    cl61_ct = numpy.count_nonzero(cl61_flags == 1)
    cl61_count.append(cl61_ct / obs_count)

    # replace -1 with None so we can perform a logical and/or
    mrr_none = numpy.where(mrr_flags == numpy.int64(-1), None, mrr_flags)
    cl61_none = numpy.where(cl61_flags == numpy.int64(-1), None, cl61_flags)

    # perform logical and/or and count occurrences of 1
    mrr_and_cl61_ct = numpy.count_nonzero(numpy.logical_and(mrr_none, cl61_none) == 1)
    mrr_AND_cl61.append(mrr_and_cl61_ct / obs_count)

    mrr_or_cl61_ct = numpy.count_nonzero(numpy.logical_or(mrr_none, cl61_none) == 1)
    mrr_OR_cl61.append(mrr_or_cl61_ct / obs_count)


if len(mrr_count) != len(fpaths):
    raise ValueError('number of counts does not match number of files')


# print out results and write into csv
N_data = len(t_start_list)

print('#Idx        t_start           t_end            Pre-gap     Duration  Post-gap  MRR_frac%  CL61_frac%  AND_frac% OR_frac%')
rows = []
row = ['#idx', 't_start', 't_end', 'pre_gap', 'duration', 'post-gap', 'mrr_frac%', 'cl61_frac%', 'mrr_and_cl_frac%', 'mrr_or_cl_frac%']
rows.append(row)
for i in range(N_data):
    if i == 0:
        pre_gap = -999.
    else:
        pre_gap = (t_start_list[i] - t_end_list[i - 1]) / numpy.timedelta64(3600, 's')
    if i == N_data - 1:
        post_gap = -999.
    else:
        post_gap = (t_start_list[i + 1] - t_end_list[i]) / numpy.timedelta64(3600, 's')
    duration = (t_end_list[i] - t_start_list[i]) / numpy.timedelta64(3600, 's')
    f_out = f' {i:3d} {t_start_list[i]} {t_end_list[i]} {pre_gap:10.3f} {duration:10.3f} {post_gap:10.3f} ' \
            f'{mrr_count[i] * 100:10.3f} {cl61_count[i] * 100:10.3f} {mrr_AND_cl61[i] * 100:10.3f} {mrr_OR_cl61[i] * 100:10.3f} '
    rows.append(f_out.split())
    print(f_out)

# write csv with above results
with open('events_w_frac_of_occurance.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

