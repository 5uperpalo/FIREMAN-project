"""Main function for UCI letter and spam datasets.
"""

# Necessary packages
from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
from impute_data import impute_data

from utils import binary_sampler, rmse_loss


def main(args):
    """Main function for UCI letter and spam datasets.

    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    """

    data_name = args.data_name
    miss_rate = args.miss_rate
    data_path = args.data_path

    gain_parameters = {
        "batch_size": args.batch_size,
        "hint_rate": args.hint_rate,
        "alpha": args.alpha,
        "miss_rate": miss_rate,
        "data_name": data_name,
        "data_path": data_path,
        "iterations": args.iterations,
    }

    batch_size = int(gain_parameters["batch_size"])

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

    impute_data(fnames, gain_parameters)


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
    parser.add_argument("--miss_rate", help="missing data probability", default=0.2, type=float)
    parser.add_argument(
        "--batch_size",
        help="the number of samples in mini-batch",
        default=128,
        type=int,
    )
    parser.add_argument("--hint_rate", help="hint probability", default=0.9, type=float)
    parser.add_argument("--alpha", help="hyperparameter", default=100, type=float)
    parser.add_argument("--iterations", help="number of training interations", default=10000, type=int)
    parser.add_argument("--data_path", default=1, type=int)

    args = parser.parse_args()

    # Calls main function
    main(args)
    # np.savetxt("imputed_data.csv", imputed_data, delimiter=",")
