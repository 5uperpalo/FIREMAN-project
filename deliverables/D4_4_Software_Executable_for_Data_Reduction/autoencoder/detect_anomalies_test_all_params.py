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
        default=20,
        type=int,
    )
    parser.add_argument("--batch_size", help="batch size", default=512, type=int)
    parser.add_argument("--lags", help="pnumber of lags", default=1, type=int)
    parser.add_argument("--loss_threshold", help="loss_threshold", default=0.113, type=float)
    parser.add_argument(
        "--perc_of_data_withing_bound",
        help="perc_of_data_withing_bound",
        default=0.95,
        type=float,
    )
    parser.add_argument("--perc_of_outputs", help="perc_of_outputs", default=0.75, type=float)
    args = parser.parse_args()

    data_name = str(args.data_name)
    skip_rows = int(args.skip_rows)
    batch_size = int(args.batch_size)
    lags = int(args.lags)
    loss_threshold = float(args.loss_threshold)
    perc_of_data_withing_bound = float(args.perc_of_data_withing_bound)
    perc_of_outputs = float(args.perc_of_outputs)

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

    perc_of_data_withing_bound = [i for i in np.arange(0.98, 0, -0.05)]
    perc_of_outputs = [i for i in np.arange(0, 0.98, 0.05)]
    for perc_of_data_withing_bound_i in perc_of_data_withing_bound:
        loss_threshold_arr = train_loss_fast2(model, model_parameters, perc_of_data_withing_bound_i)

        excluded_faults = [3, 9, 15]
        faults_list = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
        false_alarm_ratio = []
        for perc_of_outputs_i in perc_of_outputs:
            for fault_no in faults_list:
                if fault_no in excluded_faults:
                    continue
                if fault_no == 0:
                    print(
                        perc_of_data_withing_bound_i,
                        perc_of_outputs_i,
                        0,
                        detect_anomalies2(
                            model,
                            model_parameters,
                            fault_no,
                            loss_threshold_arr,
                            perc_of_outputs_i,
                        ),
                    )
                else:
                    false_alarm_ratio.append(
                        detect_anomalies2(
                            model,
                            model_parameters,
                            fault_no,
                            loss_threshold_arr,
                            perc_of_outputs_i,
                        )
                    )
            print(
                perc_of_data_withing_bound_i,
                perc_of_outputs_i,
                np.mean(false_alarm_ratio),
            )
            # print(false_alarm_ratio)
            print("**********************")
