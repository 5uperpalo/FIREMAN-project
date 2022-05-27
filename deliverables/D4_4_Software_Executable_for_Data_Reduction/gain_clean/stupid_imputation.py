"""GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

# Necessary packages
import numpy as np


def binary_sampler(p, rows, cols):
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def stupid_imputation(data_x, miss_data_x, window):
    # data_m = 1-np.isnan(data_x)
    for row in range(len(miss_data_x)):
        for col in range(len(miss_data_x[row])):
            if np.isnan(miss_data_x[row][col]):
                if row == 0:
                    miss_data_x[row][col] = 0
                    logging.debug("************************")
                    logging.debug(str(miss_data_x[row][col]))
                    logging.debug(str(data_x[row][col]))
                    logging.debug("************************")
                elif window > row:
                    col_val = []
                    for w in range(1, row + 1):
                        col_val.append(miss_data_x[row - w][col])
                    miss_data_x[row][col] = np.mean(col_val)
                    logging.debug("************************")
                    logging.debug(str(miss_data_x[row][col]))
                    logging.debug(str(data_x[row][col]))
                    logging.debug("************************")
                else:
                    col_val = []
                    for w in range(1, window + 1):
                        col_val.append(miss_data_x[row - w][col])
                    miss_data_x[row][col] = np.mean(col_val)
                    logging.debug("************************")
                    logging.debug(str(miss_data_x[row][col]))
                    logging.debug(str(data_x[row][col]))
                    logging.debug("************************")


def impute_data(fnames, imputation_parameters):
    impute_logging_file = "debug/" + str(imputation_parameters["data_path"]) + ".log"
    logging.basicConfig(filename=impute_logging_file, level=logging.DEBUG)

    miss_rate = float(imputation_parameters["miss_rate"])

    for f in fnames:
        logging.debug("************************")
        logging.debug(str(f))
        data_x = np.loadtxt(f, delimiter=",", skiprows=1)
        # Parameters
        no, dim = data_x.shape

        # Introduce missing data
        data_m = binary_sampler(1 - miss_rate, no, dim)
        miss_data_x = data_x.copy()
        miss_data_x[data_m == 0] = np.nan

        stupid_imputation(data_x, miss_data_x, imputation_parameters["window"])

        out_file_name = "./results/" + str(imputation_parameters["data_name"]) + "/" + str(os.path.basename(f))
        with open(out_file_name, "w") as f:
            np.savetxt(f, miss_data_x, delimiter=",")


def main(args):
    """Main function for UCI letter and spam datasets.

    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch:size: batch size

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    """

    data_name = args.data_name
    miss_rate = args.miss_rate
    data_path = args.data_path
    window = args.window

    imputation_parameters = {
        "miss_rate": miss_rate,
        "data_name": data_name,
        "data_path": data_path,
        "window": window,
    }

    # Load data and introduce missingness
    from glob import glob

    data_arr = []
    if data_name == "fault_free_training":
        fnames = glob("data/fault_free_training/fault_free_training_p*/*")
    elif data_name == "fault_free_testing":
        fnames = glob("data/fault_free_testing/fault_free_testing_p*/*")
    elif data_name == "faulty_training":
        data_arr = []
        fnames0 = glob("data/fault_free_training/fault_free_training_p*/fault*")
        fnames1 = glob("data/faulty_training/faulty_training_1/faulty_training_1_p*/fault*")
        fnames2 = glob("data/faulty_training/faulty_training_2/faulty_training_2_p*/fault*")
        fnames3 = glob("data/faulty_training/faulty_training_3/faulty_training_3_p*/fault*")
        fnames4 = glob("data/faulty_training/faulty_training_4/faulty_training_4_p*/fault*")
        fnames5 = glob("data/faulty_training/faulty_training_5/faulty_training_5_p*/fault*")
        fnames6 = glob("data/faulty_training/faulty_training_6/faulty_training_6_p*/fault*")
        fnames7 = glob("data/faulty_training/faulty_training_7/faulty_training_7_p*/fault*")
        fnames8 = glob("data/faulty_training/faulty_training_8/faulty_training_8_p*/fault*")
        fnames9 = glob("data/faulty_training/faulty_training_9/faulty_training_9_p*/fault*")
        fnames10 = glob("data/faulty_training/faulty_training_10/faulty_training_10_p*/fault*")
        fnames11 = glob("data/faulty_training/faulty_training_11/faulty_training_11_p*/fault*")
        fnames12 = glob("data/faulty_training/faulty_training_12/faulty_training_12_p*/fault*")
        fnames13 = glob("data/faulty_training/faulty_training_13/faulty_training_13_p*/fault*")
        fnames14 = glob("data/faulty_training/faulty_training_14/faulty_training_14_p*/fault*")
        fnames15 = glob("data/faulty_training/faulty_training_15/faulty_training_15_p*/fault*")
        fnames16 = glob("data/faulty_training/faulty_training_16/faulty_training_16_p*/fault*")
        fnames17 = glob("data/faulty_training/faulty_training_17/faulty_training_17_p*/fault*")
        fnames18 = glob("data/faulty_training/faulty_training_18/faulty_training_18_p*/fault*")
        fnames19 = glob("data/faulty_training/faulty_training_19/faulty_training_19_p*/fault*")
        fnames20 = glob("data/faulty_training/faulty_training_20/faulty_training_20_p*/fault*")
        fnames = [
            y
            for x in [
                fnames0,
                fnames1,
                fnames2,
                fnames3,
                fnames4,
                fnames5,
                fnames6,
                fnames7,
                fnames8,
                fnames9,
                fnames10,
                fnames11,
                fnames12,
                fnames13,
                fnames14,
                fnames15,
                fnames16,
                fnames17,
                fnames18,
                fnames19,
                fnames20,
            ]
            for y in x
        ]
        # fnames = [y for x in [fnames3] for y in x]
    elif data_name == "faulty_testing":
        data_arr = []
        if data_path == 0:
            data_path_whole = "data/fault_free_testing/fault_free_testing_p*/fault*"
        else:
            data_path_whole = (
                "data/faulty_testing/faulty_testing_" + str(data_path) + "/faulty_testing_" + str(data_path) + "_p*/fault*"
            )
        fnames0 = glob(data_path_whole)
        fnames = [y for x in [fnames0] for y in x]

    impute_data(fnames, imputation_parameters)


if __name__ == "__main__":

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        choices=[
            "fault_free_training",
            "fault_free_testing",
            "faulty_training",
            "faulty_testing",
        ],
        default="spam",
        type=str,
    )
    parser.add_argument("--miss_rate", help="missing data probability", default=0.1, type=float)
    parser.add_argument("--data_path", default=1, type=int)
    parser.add_argument("--window", default=1, type=int)

    args = parser.parse_args()

    # Calls main function
    main(args)
