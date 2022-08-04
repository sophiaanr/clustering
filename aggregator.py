#!/usr/bin/env python3


# This code does an aggregation of a list of "events" with specified start and end times
# The aggregation is based on the duration of an event (it has to be long enough to be significant)
# and on the gaps between that significant event and the nearest events before and after it.
# If the gaps are small enough, the preceeding and succeeding events are joined to the
# significant event to form an aggregate event.

# The aggregate code can be run iteratively on the event time data until the event time data ceases
# to change (see the example code below after the aggregate() routine)

# Norm Wood, norman.wood@ssec.wisc.edu
import h5py
import numpy

from event_detection import write_csv


def unix_to_datetime64(unix_time):
    return unix_time * numpy.timedelta64(1, 's') + numpy.datetime64('1970-01-01T00:00:00Z')


def main():
    with open('snow_events_800_2.csv', 'r') as f_in:
        lines = f_in.readlines()

    t_start_list = []
    t_end_list = []

    # for line in lines[1:]:
    #     parts = line.split(',')
    #     t_start_list.append(numpy.datetime64(parts[1]))
    #     t_end_list.append(numpy.datetime64(parts[2]))

    with h5py.File('detection_flags_mrr_bin5_m12_cl61_bin6_m6.h5') as f:
        time = f['time'][:]

    t_start_list = unix_to_datetime64(time[:-1])
    t_end_list = [x + numpy.timedelta64(10, 's') for x in t_start_list]

    time_data = numpy.array([t_start_list, t_end_list])

    N_data = time_data.shape[1]
    dt_gap_thresh = numpy.timedelta64(5400, 's')
    dt_duration_min = numpy.timedelta64(1800, 's')
    N_data_prior = N_data
    count = 0
    print(N_data)
    while True:
        time_data_agg = aggregate(time_data, dt_gap_thresh, dt_duration_min)
        N_data_current = time_data_agg.shape[1]
        if N_data_current == N_data_prior:
            break
        else:
            N_data_prior = N_data_current
            time_data = time_data_agg
            count += 1
            print(count, N_data_current)

    durations = time_data[1, :] - time_data[0, :]

    # rows = []
    for idx in range(N_data_current):
        if idx < N_data_current - 1:
            gap = (time_data[0, idx + 1] - time_data[1, idx]) / numpy.timedelta64(3600, 's')
        else:
            gap = -999.

        row = f'{idx} {time_data[0, idx]} {time_data[1, idx]} {durations[idx] / numpy.timedelta64(3600, "s"):.3f} {gap:.3f}'
        print(row)
        # rows.append(row.split())
        # print('%4d %s %s %12.3f %12.3f' % (
        #     idx, time_data[0, idx], time_data[1, idx], durations[idx] / numpy.timedelta64(3600, 's'), gap))
    # f_out = 'aggregated_snow_events_800_2.csv'
    # write_csv(rows, f_out)


def aggregate(time_data, dt_gap_thresh, dt_duration_min):
    N_data = time_data.shape[1]
    durations = time_data[1, :] - time_data[0, :]
    idx_targets = numpy.argsort(durations)
    idx_processed_list = []
    t_start_list = []
    t_end_list = []
    for idx in idx_targets[::-1]:
        if idx not in idx_processed_list:
            idx_processed_list.append(idx)
            if durations[idx] > dt_duration_min:
                # Significant event, process it
                # Find the nearest-neighbor episodes and see if they are within dt_gap_thresh
                # Check before the current episode
                # Difference the start time of the current episode with the end times of the preceeding episodes
                idx_adjacent = idx - 1
                if idx_adjacent >= 0:
                    if idx_adjacent not in idx_processed_list:
                        adj_diff = numpy.abs(time_data[0, idx] - time_data[1, idx_adjacent])
                        if adj_diff < dt_gap_thresh:
                            # Consume this episode
                            action_string_start = 'Consumed %s at start' % (idx_adjacent)
                            t_start_list.append(time_data[0, idx_adjacent])
                            idx_processed_list.append(idx_adjacent)
                        else:
                            action_string_start = 'Unchanged at start'
                            t_start_list.append(time_data[0, idx])
                    else:
                        action_string_start = 'Unchanged at start'
                        t_start_list.append(time_data[0, idx])
                else:
                    action_string_start = 'Unchanged_at start'
                    t_start_list.append(time_data[0, idx])

                # Check after the current episode
                idx_adjacent = idx + 1
                if idx_adjacent < N_data:
                    if idx_adjacent not in idx_processed_list:
                        adj_diff = numpy.abs(time_data[1, idx] - time_data[0, idx_adjacent])
                        if adj_diff < dt_gap_thresh:
                            # Consume this episode
                            action_string_end = 'Consumed %s at end' % (idx_adjacent)
                            t_end_list.append(time_data[1, idx_adjacent])
                            idx_processed_list.append(idx_adjacent)
                        else:
                            action_string_end = 'Unchanged at end'
                            t_end_list.append(time_data[1, idx])
                    else:
                        action_string_end = 'Unchanged at end'
                        t_end_list.append(time_data[1, idx])
                else:
                    action_string_end = 'Unchanged at end'
                    t_end_list.append(time_data[1, idx])
        else:
            pass

    time_data_new = numpy.array([t_start_list, t_end_list])
    # Rearrange into increasing t_start order
    time_data_new_sorted = time_data_new[:, time_data_new[0, :].argsort()]
    return time_data_new_sorted


if __name__ == '__main__':
    main()