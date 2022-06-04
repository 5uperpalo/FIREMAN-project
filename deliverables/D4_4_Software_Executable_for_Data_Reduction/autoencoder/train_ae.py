import argparse
import os

from a_e_model import *

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        choices=[
            "fault_free_training",
            "faulty_training",
            "faulty_testing",
            "fault_free_training_test",
        ],
        default="fault_free_training_test",
        type=str,
    )
    parser.add_argument(
        "--skip_rows",
        help="how many rows to skip in the data file",
        default=1,
        type=int,
    )
    parser.add_argument("--batch_size", help="batch size", default=512, type=int)
    parser.add_argument("--iterations", help="iterations", default=2000, type=int)
    parser.add_argument("--lags", help="pnumber of lags", default=1, type=int)
    args = parser.parse_args()

    data_name = str(args.data_name)
    skip_rows = int(args.skip_rows)
    batch_size = int(args.batch_size)
    iterations = int(args.iterations)
    lags = int(args.lags)

    ## **Defining folder paths**
    model_dir = "model/weights/"
    normalization_params_dir = "model/normalization_parameters/"
    debug_dir = "debug/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(normalization_params_dir):
        os.makedirs(normalization_params_dir)
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # Load data
    norm_data, norm_parameters = data_loader_autoencoder(data_name, lags, skip_rows, normalization_params_dir)

    model_parameters = {
        "batch_size": batch_size,
        "norm_parameters": norm_parameters,
        "iterations": iterations,
    }

    # train autoencoder
    train_autoencoder(norm_data, model_parameters, model_dir)