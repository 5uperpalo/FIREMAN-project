import logging

import numpy as np
import stumpy
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

from src import peak_detection

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger("TimeSeries")
logger.setLevel(logging.INFO)


def init(x, lag, threshold, influence):
    '''
    Smoothed z-score algorithm
    Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
    '''

    labels = np.zeros(lag)
    filtered_y = np.array(x[0:lag])
    avg_filter = np.zeros(lag)
    std_filter = np.zeros(lag)
    var_filter = np.zeros(lag)

    avg_filter[lag - 1] = np.mean(x[0:lag])
    std_filter[lag - 1] = np.std(x[0:lag])
    var_filter[lag - 1] = np.var(x[0:lag])

    return dict(avg=avg_filter[lag - 1],
                var=var_filter[lag - 1],
                std=std_filter[lag - 1],
                filtered_y=filtered_y,
                labels=labels)


def add(result, single_value, lag, threshold, influence):
    previous_avg = result['avg']
    previous_var = result['var']
    previous_std = result['std']
    filtered_y = result['filtered_y']
    labels = result['labels']

    if abs(single_value - previous_avg) > threshold * previous_std:
        if single_value > previous_avg:
            labels = np.append(labels, 1)
        else:
            labels = np.append(labels, -1)

        # calculate the new filtered element using the influence factor
        filtered_y = np.append(filtered_y, influence * single_value
                               + (1 - influence) * filtered_y[-1])
    else:
        labels = np.append(labels, 0)
        filtered_y = np.append(filtered_y, single_value)

    # update avg as sum of the previuos avg + the lag * (the new calculated item - calculated item at position (i - lag))
    current_avg_filter = previous_avg + 1. / lag * (filtered_y[-1]
                                                    - filtered_y[len(filtered_y) - lag - 1])

    # update variance as the previuos element variance + 1 / lag * new recalculated item - the previous avg -
    current_var_filter = previous_var + 1. / lag * ((filtered_y[-1]
                                                     - previous_avg) ** 2 - (filtered_y[len(filtered_y) - 1
                                                                                        - lag] - previous_avg) ** 2 - (
                                                                filtered_y[-1]
                                                                - filtered_y[len(
                                                            filtered_y) - 1 - lag]) ** 2 / lag)  # the recalculated element at pos (lag) - avg of the previuos - new recalculated element - recalculated element at lag pos ....

    # calculate standard deviation for current element as sqrt (current variance)
    current_std_filter = np.sqrt(current_var_filter)

    return dict(avg=current_avg_filter,
                var=current_var_filter,
                std=current_std_filter,
                filtered_y=filtered_y[1:],
                labels=labels)


lag = 30
threshold = 5
influence = 0


class OutlierModel:
    """Object used to define outlier network designs"""

    def __init__(
            self,
            m: int = 15,
            std_dev: int = 3,
            time_series=None,
            egress=True,
            lag: float = 100,
            threshold: float = 5,
            influence: float = 0.5
    ):
        self.m = m
        if time_series is None:
            time_series = np.zeros(self.m * 4)
        self.time_series = time_series
        self.stream = stumpy.stumpi(self.time_series, m, egress=egress)
        self.lastMaxIndex = -1
        self.anomalies = []
        self.std_dev = std_dev
        self.count = 0
        self.max_std_dev = []
        self.max_mean = []
        self.max_val = []
        self.comparisson = []
        # Settings: lag = 30, threshold = 5, influence = 0
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.peak_detection = peak_detection.real_time_peak_detection(np.zeros(self.lag),self.lag,self.threshold,self.influence)
        self.calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=len(self.time_series) * 5)

    def __repr__(self):
        return str(self.__dict__)

    def train(self):
        pass

    def train_one(self, data):
        self.stream.update(data)

    def predict_one(self, index):
        self.count += 1
        max_mp = np.round(self.stream.P_.max(), 4)
        mean_mp = np.round(self.stream.P_.mean(), 4)
        std_dev_mp = np.round(self.stream.P_.std(), 4)
        self.max_val.append(max_mp)
        self.max_mean.append(mean_mp)
        self.max_std_dev.append(std_dev_mp)

        max_index = np.argwhere(self.stream.P_ == self.stream.P_.max()).flatten()[0]
        anomaly = False
        true_anomaly = False
        metric = self.peak_detection.thresholding_algo(max_mp)
        # if self.count >= self.lag:
        #     local_window = self.max_val[-self.lag:]
        #     mean_local = np.average(local_window)
        #     std_dev_local = np.std(local_window)
        #     metric = (max_mp - mean_local) / std_dev_local
        # else:
        #     metric = 0
        # self.peak_detection = add(self.peak_detection, max_index, self.lag, self.threshold, self.peak_detection)

        self.comparisson.append(metric)

        if metric == 1:
            anomaly = True
            # if self.lastMaxIndex >= 0:
            #     if self.lastMaxIndex != max_index:
            #         anomaly = True
            # else:
            #     anomaly = True
        if anomaly and not self.is_warming_up() and not self.recent_fault(index):
            true_anomaly = True
            self.anomalies.append(index)
            logger.warning(f" Anomaly at Global index: {index}, local index: {max_index}")
        self.lastMaxIndex = max_index
        return true_anomaly

    def is_warming_up(self):
        return self.count <= self.m

    def recent_fault(self, index):
        return index < (self.anomalies[-1] + self.m) if self.anomalies else False
