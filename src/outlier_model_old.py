import logging

import numpy as np
import stumpy
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

from src import peak_detection

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger("TimeSeries")
logger.setLevel(logging.INFO)


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
        self.avgFilter = []
        self.stdFilter = []
        # Settings: lag = 30, threshold = 5, influence = 0
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.peak_detection = peak_detection.real_time_peak_detection(np.zeros(self.lag), self.lag, self.threshold,
                                                                      self.influence)
        logger.warning(f"Lag: {self.lag} Threshold: {self.threshold} Influence: {self.influence}")
        # self.calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=len(self.time_series) * 5)

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

        # max_index = np.argwhere(self.stream.P_ == self.stream.P_.max()).flatten()[0]
        anomaly = False
        true_anomaly = False
        metric = self.peak_detection.thresholding_algo(max_mp)
        self.stdFilter.append(metric['stdFilter'])
        self.avgFilter.append(metric['avgFilter'])
        # if self.count >= self.lag:
        #     local_window = self.max_val[-self.lag:]
        #     mean_local = np.average(local_window)
        #     std_dev_local = np.std(local_window)
        #     metric = (max_mp - mean_local) / std_dev_local
        # else:
        #     metric = 0
        # self.peak_detection = add(self.peak_detection, max_index, self.lag, self.threshold, self.peak_detection)

        self.comparisson.append(metric['signal'])

        if metric['signal'] == 1:
            anomaly = True
            # if self.lastMaxIndex >= 0:
            #     if self.lastMaxIndex != max_index:
            #         anomaly = True
            # else:
            #     anomaly = True
        if anomaly and not self.is_warming_up() and not self.recent_fault(index):
            true_anomaly = True
            self.anomalies.append(index)
            logger.warning(f" Anomaly at Global index: {index}")
        # self.lastMaxIndex = max_index
        return true_anomaly

    def is_warming_up(self):
        return self.count <= self.m

    def recent_fault(self, index):
        return index < (self.anomalies[-1] + self.m) if self.anomalies else False
