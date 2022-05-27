"""Data loader for UCI letter, spam and MNIST datasets.
"""

import os

# Necessary packages
import numpy as np

from utils import (
    binary_sampler,
    binary_sampler_with_lags,
    normalization,
    sample_always_specific_sensor,
)

# from keras.datasets import mnist


def introduce_delayed_measurements_mine(data_arr, max_delay):
    # logging.basicConfig(filename='gain.log',level=logging.DEBUG)
    new_return_data_arr = []
    for arr in data_arr:
        one_matrix_arr = []
        for line_id in range(len(arr)):
            new_arr = []
            for j in range(1, max_delay + 1):
                for i in range(len(arr[line_id])):
                    if j <= line_id:
                        # new_arr.append(arr[line_id][i] - arr[line_id-j][i])
                        new_arr.append(arr[line_id - j][i])
                    else:
                        # new_arr.append(arr[line_id][i])
                        new_arr.append(0)
            one_matrix_arr.append(new_arr)
        # logging.debug("******************")
        # logging.debug(str(np.array(one_matrix_arr).shape))
        arr = np.append(np.array(arr), np.array(one_matrix_arr), 1)
        # logging.debug(str(arr.shape))
        new_return_data_arr.append(arr)

    return new_return_data_arr


def introduce_delayed_measurements(data_arr, max_delay):
    new_return_data_arr = []
    data_arr = np.array(data_arr)
    for arr in data_arr:
        mat_past = arr
        for i in range(1, max_delay):
            mat_past = np.concatenate((arr[i:, :], mat_past[:-1, :]), axis=1)
        new_return_data_arr.append(mat_past)

    return new_return_data_arr


def data_loader(data_name, miss_rate, skip_r=1, alpha=100, dir_name="0"):
    """Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    """

    # Load data
    if data_name in ["letter", "spam"]:
        file_name = "data/" + data_name + ".csv"
        data_x = np.loadtxt(file_name, delimiter=",", skiprows=skip_r)
    # elif data_name == 'mnist':
    # (data_x, _), _ = mnist.load_data()
    # data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
    elif data_name == "tennesee_dat":
        file_name = "data/" + data_name + ".csv"
        data_x = np.loadtxt(file_name, delimiter=",")
    elif data_name == "fault_free_training":
        from glob import glob

        data_arr = []
        fnames = glob("data/fault_free_training/fault_free_training_p*/*")
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_x = np.concatenate(data_arr)
    elif data_name == "faulty_training":
        from glob import glob

        data_arr = []
        # fnames0 = glob('data/fault_free_training/fault_free_training_p*/fault*')
        fnames1 = glob("data/faulty_training/faulty_training_1/faulty_training_1_p*/fault*")
        fnames2 = glob("data/faulty_training/faulty_training_2/faulty_training_2_p*/fault*")
        # fnames3 = glob('data/faulty_training/faulty_training_3/faulty_training_3_p*/fault*')
        fnames4 = glob("data/faulty_training/faulty_training_4/faulty_training_4_p*/fault*")
        fnames5 = glob("data/faulty_training/faulty_training_5/faulty_training_5_p*/fault*")
        # fnames6 = glob('data/faulty_training/faulty_training_6/faulty_training_6_p*/fault*')
        fnames7 = glob("data/faulty_training/faulty_training_7/faulty_training_7_p*/fault*")
        fnames8 = glob("data/faulty_training/faulty_training_8/faulty_training_8_p*/fault*")
        # fnames9 = glob('data/faulty_training/faulty_training_9/faulty_training_9_p*/fault*')
        fnames10 = glob("data/faulty_training/faulty_training_10/faulty_training_10_p*/fault*")
        fnames11 = glob("data/faulty_training/faulty_training_11/faulty_training_11_p*/fault*")
        fnames12 = glob("data/faulty_training/faulty_training_12/faulty_training_12_p*/fault*")
        fnames13 = glob("data/faulty_training/faulty_training_13/faulty_training_13_p*/fault*")
        fnames14 = glob("data/faulty_training/faulty_training_14/faulty_training_14_p*/fault*")
        # fnames15 = glob('data/faulty_training/faulty_training_15/faulty_training_15_p*/fault*')
        fnames16 = glob("data/faulty_training/faulty_training_16/faulty_training_16_p*/fault*")
        fnames17 = glob("data/faulty_training/faulty_training_17/faulty_training_17_p*/fault*")
        # fnames18 = glob('data/faulty_training/faulty_training_18/faulty_training_18_p*/fault*')
        fnames19 = glob("data/faulty_training/faulty_training_19/faulty_training_19_p*/fault*")
        fnames20 = glob("data/faulty_training/faulty_training_20/faulty_training_20_p*/fault*")
        # fnames = [y for x in [fnames0, fnames1, fnames2, fnames3, fnames4, fnames5, fnames6, fnames7, fnames8, fnames9, fnames10, fnames11, fnames12, fnames13, fnames14, fnames15, fnames16, fnames17, fnames18, fnames19, fnames20] for y in x]
        # fnames = [y for x in [fnames0, fnames1, fnames2, fnames4, fnames5, fnames6, fnames7, fnames8, fnames10, fnames11, fnames12, fnames13, fnames14, fnames16, fnames17, fnames18, fnames19, fnames20] for y in x]
        fnames = [
            y
            for x in [
                fnames1,
                fnames2,
                fnames4,
                fnames5,
                fnames7,
                fnames8,
                fnames10,
                fnames11,
                fnames12,
                fnames13,
                fnames14,
                fnames16,
                fnames17,
                fnames19,
                fnames20,
            ]
            for y in x
        ]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        # data_arr = introduce_delayed_measurements(data_arr, 11)
        data_arr = introduce_delayed_measurements(data_arr, 11)
        data_x = np.concatenate(data_arr)
    elif data_name == "faulty_training_test":
        from glob import glob

        data_arr = []
        fnames0 = glob("data/fault_free_training/fault_free_training_p*/fault*")
        fnames = [y for x in [fnames0] for y in x]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_arr = introduce_delayed_measurements(data_arr, 4)
        data_x = np.concatenate(data_arr)

    # Parameters
    no, dim = data_x.shape

    # even though i decided to use lags (the missing data values should be intorduced only to the features themselves (that is why orig_features is fixed in this script to 52 (this does not have to propagate to other scripts)))
    orig_features = 52

    # # Normalization
    norm_data, norm_parameters = normalization(data_x)
    # norm_data_x = np.nan_to_num(norm_data, 0)
    folder_name = "normalization_params/" + "alpha_" + str(alpha) + "_p" + str(dir_name) + "/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.savetxt(folder_name + "min_val", norm_parameters["min_val"], delimiter=",")
    np.savetxt(folder_name + "max_val", norm_parameters["max_val"], delimiter=",")

    # # Introduce missing data
    # # data_m = binary_sampler(1-miss_rate, no, dim)
    # data_m = binary_sampler_with_lags(1-miss_rate, no, dim, orig_features)
    # # data_m = sample_always_specific_sensor(no, dim, 49)
    # # miss_data_x = data_x.copy()
    # miss_data_x = norm_data.copy()
    # miss_data_x[data_m == 0] = np.nan

    # return data_x, miss_data_x, data_m, norm_parameters
    return norm_data, norm_parameters
