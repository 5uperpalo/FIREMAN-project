import argparse
import os

from a_e_model import *

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", choices=["gan", "stupid"], default="gan", type=str)
    parser.add_argument(
        "--skip_rows",
        help="how many rows to skip in the data file",
        default=160,
        type=int,
    )
    parser.add_argument("--batch_size", help="batch size", default=512, type=int)
    parser.add_argument("--lags", help="pnumber of lags", default=1, type=int)
    parser.add_argument("--loss_threshold", help="loss_threshold", default=0.113, type=float)
    args = parser.parse_args()

    data_set = str(args.data_set)
    skip_rows = int(args.skip_rows)
    batch_size = int(args.batch_size)
    lags = int(args.lags)
    loss_threshold = float(args.loss_threshold)

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
    # norm_data, norm_parameters = data_loader_autoencoder(data_name, lags, skip_rows, normalization_params_dir)

    # train autoencoder
    model, norm_parameters = build_model(model_dir, normalization_params_dir)

    model_parameters = {
        "batch_size": batch_size,
        "norm_parameters": norm_parameters,
        "lags": lags,
    }

    excluded_faults = [3, 9, 15]
    # faults_list = [0,1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20]
    faults_list = [0, 1]
    # faults_list = [1]
    faulty_sensor_arr = [0, 8, 12, 17, 18, 20, 21, 43, 44, 49, 50, 51]
    # faulty_sensor_arr = [0]
    false_alarm_ratio = []
    for fault_no in faults_list:
        if fault_no == 0:
            print(
                0,
                detect_imputed_anomalies(model, model_parameters, 0, fault_no, loss_threshold, data_set),
            )
        else:
            for faulty_sensor in faulty_sensor_arr:
                false_alarm_ratio.append(
                    detect_imputed_anomalies(
                        model,
                        model_parameters,
                        faulty_sensor,
                        fault_no,
                        loss_threshold,
                        data_set,
                    )
                )
    print(np.mean(false_alarm_ratio))
    print(false_alarm_ratio)
