import logging
import os

import numpy as np
import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, dim):
        super(AE, self).__init__()
        input_s = dim
        output_s = 52
        # next_s = [300, 200, 100, 80, 52, 40]
        # 12_layes_10_lags_model1
        # next_s = [400, 300, 200, 100, 52, 32]
        # chosen
        next_s = [52, 52, 48, 47, 46, 45]
        # better numbers
        # next_s = [52, 52, 48, 45, 43, 42]
        # ok numbers
        # next_s = [48, 40, 35, 30, 25, 20]
        # 12_layes_10_lags_model1
        self.enc = nn.Sequential(
            nn.Linear(input_s, next_s[0]),
            nn.ReLU(),
            nn.Linear(next_s[0], next_s[1]),
            nn.ReLU(),
            nn.Linear(next_s[1], next_s[2]),
            nn.ReLU(),
            nn.Linear(next_s[2], next_s[3]),
            nn.ReLU(),
            nn.Linear(next_s[3], next_s[4]),
            nn.ReLU(),
            nn.Linear(next_s[4], next_s[5]),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(next_s[5], next_s[4]),
            nn.ReLU(),
            nn.Linear(next_s[4], next_s[3]),
            nn.ReLU(),
            nn.Linear(next_s[3], next_s[2]),
            nn.ReLU(),
            nn.Linear(next_s[2], next_s[1]),
            nn.ReLU(),
            nn.Linear(next_s[1], next_s[0]),
            nn.ReLU(),
            nn.Linear(next_s[0], output_s),
            nn.ReLU(),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode


def introduce_delayed_measurements(data_arr, max_delay):
    new_return_data_arr = []
    data_arr = np.array(data_arr)
    for arr in data_arr:
        mat_past = arr
        for i in range(1, max_delay):
            mat_past = np.concatenate((arr[i:, :], mat_past[:-1, :]), axis=1)
        new_return_data_arr.append(mat_past)

    return new_return_data_arr


def normalization(data):
    """Normalize data in [0, 1] range.
    Args:
    - data: original data
    Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)

    # For each dimension
    for i in range(dim):
        min_val[i] = np.nanmin(norm_data[:, i])
        max_val[i] = np.nanmax(norm_data[:, i])
        norm_data[:, i] = (norm_data[:, i] - min_val[i]) / (max_val[i] - min_val[i] + 1e-6)

    # Return norm_parameters for renormalization
    norm_parameters = {"min_val": min_val, "max_val": max_val}

    return norm_data, norm_parameters


def data_loader_autoencoder_use_my_own_norm(
    data_name,
    norm_parameters,
    no_lags,
    skip_r=1,
    normalization_params_dir="model/normalization_parameters/",
):
    # Load data
    if data_name == "fault_free_training":
        from glob import glob

        data_arr = []
        # fnames0 = glob('data/fault_free_training/fault_free_training_p*/*')
        fnames0 = glob("data/fault_free_training/smoothed_fault_free_training/fault*")
        fnames = [y for x in [fnames0] for y in x]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
    elif data_name == "faulty_training":
        from glob import glob

        data_arr = []
        fnames0 = glob("data/fault_free_training/smoothed_fault_free_training/fault*")
        fnames1 = glob("data/faulty_training/faulty_training_1/smoothed_faulty_training_1/fault*")
        fnames2 = glob("data/faulty_training/faulty_training_2/smoothed_faulty_training_2/fault*")
        fnames4 = glob("data/faulty_training/faulty_training_4/smoothed_faulty_training_4/fault*")
        fnames5 = glob("data/faulty_training/faulty_training_5/smoothed_faulty_training_5/fault*")
        fnames6 = glob("data/faulty_training/faulty_training_6/smoothed_faulty_training_6/fault*")
        fnames7 = glob("data/faulty_training/faulty_training_7/smoothed_faulty_training_7/fault*")
        fnames8 = glob("data/faulty_training/faulty_training_8/smoothed_faulty_training_8/fault*")
        fnames10 = glob("data/faulty_training/faulty_training_10/smoothed_faulty_training_10/fault*")
        fnames11 = glob("data/faulty_training/faulty_training_11/smoothed_faulty_training_11/fault*")
        fnames12 = glob("data/faulty_training/faulty_training_12/smoothed_faulty_training_12/fault*")
        fnames13 = glob("data/faulty_training/faulty_training_13/smoothed_faulty_training_13/fault*")
        fnames14 = glob("data/faulty_training/faulty_training_14/smoothed_faulty_training_14/fault*")
        fnames16 = glob("data/faulty_training/faulty_training_16/smoothed_faulty_training_16/fault*")
        fnames17 = glob("data/faulty_training/faulty_training_17/smoothed_faulty_training_17/fault*")
        fnames18 = glob("data/faulty_training/faulty_training_18/smoothed_faulty_training_18/fault*")
        fnames19 = glob("data/faulty_training/faulty_training_19/smoothed_faulty_training_19/fault*")
        fnames20 = glob("data/faulty_training/faulty_training_20/smoothed_faulty_training_20/fault*")
        fnames = [
            y
            for x in [
                fnames0,
                fnames1,
                fnames2,
                fnames4,
                fnames5,
                fnames6,
                fnames7,
                fnames8,
                fnames10,
                fnames11,
                fnames12,
                fnames13,
                fnames14,
                fnames16,
                fnames17,
                fnames18,
                fnames19,
                fnames20,
            ]
            for y in x
        ]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
    elif data_name == "fault_free_training_test":
        from glob import glob

        data_arr = []
        # fnames0 = glob('data/fault_free_training/fault_free_training_p1/*')
        fnames0 = glob("data/fault_free_training/fault_free_training_p*/*")
        fnames = [y for x in [fnames0] for y in x]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)

    # Parameters
    no, dim = data_x.shape

    orig_features = 52

    # # Normalization
    norm_data = normalize(data_x, norm_parameters)

    return norm_data


def data_loader_autoencoder(
    data_name,
    no_lags,
    skip_r=1,
    normalization_params_dir="model/normalization_parameters/",
):
    # Load data
    if data_name == "fault_free_training":
        from glob import glob

        data_arr = []
        # fnames0 = glob('data/fault_free_training/fault_free_training_p*/*')
        fnames0 = glob("data/fault_free_training/smoothed_fault_free_training/fault*")
        fnames = [y for x in [fnames0] for y in x]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
    elif data_name == "faulty_training":
        from glob import glob

        data_arr = []
        fnames0 = glob("data/fault_free_training/smoothed_fault_free_training/fault*")
        fnames1 = glob("data/faulty_training/faulty_training_1/smoothed_faulty_training_1/fault*")
        fnames2 = glob("data/faulty_training/faulty_training_2/smoothed_faulty_training_2/fault*")
        fnames4 = glob("data/faulty_training/faulty_training_4/smoothed_faulty_training_4/fault*")
        fnames5 = glob("data/faulty_training/faulty_training_5/smoothed_faulty_training_5/fault*")
        fnames6 = glob("data/faulty_training/faulty_training_6/smoothed_faulty_training_6/fault*")
        fnames7 = glob("data/faulty_training/faulty_training_7/smoothed_faulty_training_7/fault*")
        fnames8 = glob("data/faulty_training/faulty_training_8/smoothed_faulty_training_8/fault*")
        fnames10 = glob("data/faulty_training/faulty_training_10/smoothed_faulty_training_10/fault*")
        fnames11 = glob("data/faulty_training/faulty_training_11/smoothed_faulty_training_11/fault*")
        fnames12 = glob("data/faulty_training/faulty_training_12/smoothed_faulty_training_12/fault*")
        fnames13 = glob("data/faulty_training/faulty_training_13/smoothed_faulty_training_13/fault*")
        fnames14 = glob("data/faulty_training/faulty_training_14/smoothed_faulty_training_14/fault*")
        fnames16 = glob("data/faulty_training/faulty_training_16/smoothed_faulty_training_16/fault*")
        fnames17 = glob("data/faulty_training/faulty_training_17/smoothed_faulty_training_17/fault*")
        fnames18 = glob("data/faulty_training/faulty_training_18/smoothed_faulty_training_18/fault*")
        fnames19 = glob("data/faulty_training/faulty_training_19/smoothed_faulty_training_19/fault*")
        fnames20 = glob("data/faulty_training/faulty_training_20/smoothed_faulty_training_20/fault*")
        fnames = [
            y
            for x in [
                fnames0,
                fnames1,
                fnames2,
                fnames4,
                fnames5,
                fnames6,
                fnames7,
                fnames8,
                fnames10,
                fnames11,
                fnames12,
                fnames13,
                fnames14,
                fnames16,
                fnames17,
                fnames18,
                fnames19,
                fnames20,
            ]
            for y in x
        ]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
    elif data_name == "fault_free_training_test":
        from glob import glob

        data_arr = []
        # fnames0 = glob('data/fault_free_training/fault_free_training_p1/*')
        fnames0 = glob("data/fault_free_training/fault_free_training_p*/*")
        fnames = [y for x in [fnames0] for y in x]
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r) for f in fnames]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)

    # Parameters
    no, dim = data_x.shape

    orig_features = 52

    # # Normalization
    norm_data, norm_parameters = normalization(data_x)
    # norm_data_x = np.nan_to_num(norm_data, 0)
    np.savetxt(normalization_params_dir + "min_val", norm_parameters["min_val"], delimiter=",")
    np.savetxt(normalization_params_dir + "max_val", norm_parameters["max_val"], delimiter=",")

    return norm_data, norm_parameters


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def continue_train_autoencoder(model, data_x, model_parameters, model_dir="model/weights/"):
    orig_features = 52
    logging.basicConfig(filename="./debug/train_ae.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    num_of_iterations = int(model_parameters["iterations"])

    no, dim = data_x.shape

    logging.debug("Num samples in dataset:")
    logging.debug(str(no))
    logging.debug(str(dim))

    # creating the model
    # model = AE(dim)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    mse_loss = nn.MSELoss(reduction="mean")

    # signaling to the model that it will be trained
    model.train()

    # Start Iterations
    for it in range(num_of_iterations):
        mb_idx = sample_idx(no, mb_size)
        X_mb = data_x[mb_idx, :]
        X_mb_orig = X_mb[:, :orig_features]

        # create tensors
        X_mb = torch.tensor(X_mb).float()
        X_mb_orig = torch.tensor(X_mb_orig).float()

        running_loss = 0.0
        # zero the parameter gradients
        optim.zero_grad()

        # forward + backward + optimize
        # sample = model(X_mb)
        # loss = mse_loss(X_mb, sample)
        sample = model(X_mb)
        loss = mse_loss(X_mb_orig, sample)
        loss.backward()
        optim.step()
        running_loss += loss.item()

        logging.debug("Iteration {} loss: {:.4}".format(it, running_loss))

    # save the model
    np.savetxt(model_dir + "AE_enc_W1", model.enc[0].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b1", model.enc[0].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W2", model.enc[2].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b2", model.enc[2].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W3", model.enc[4].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b3", model.enc[4].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W4", model.enc[6].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b4", model.enc[6].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W5", model.enc[8].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b5", model.enc[8].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W6", model.enc[10].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b6", model.enc[10].bias.data, delimiter=",")

    np.savetxt(model_dir + "AE_dec_W1", model.dec[0].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b1", model.dec[0].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W2", model.dec[2].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b2", model.dec[2].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W3", model.dec[4].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b3", model.dec[4].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W4", model.dec[6].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b4", model.dec[6].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W5", model.dec[8].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b5", model.dec[8].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W6", model.dec[10].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b6", model.dec[10].bias.data, delimiter=",")


def train_autoencoder(data_x, model_parameters, model_dir="model/weights/"):
    orig_features = 52
    logging.basicConfig(filename="./debug/train_ae.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    num_of_iterations = int(model_parameters["iterations"])

    no, dim = data_x.shape

    logging.debug("Num samples in dataset:")
    logging.debug(str(no))
    logging.debug(str(dim))

    # creating the model
    model = AE(dim)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    mse_loss = nn.MSELoss(reduction="mean")

    # signaling to the model that it will be trained
    model.train()

    # Start Iterations
    for it in range(num_of_iterations):
        mb_idx = sample_idx(no, mb_size)
        X_mb = data_x[mb_idx, :]
        X_mb_orig = X_mb[:, :orig_features]

        # create tensors
        X_mb = torch.tensor(X_mb).float()
        X_mb_orig = torch.tensor(X_mb_orig).float()

        running_loss = 0.0
        # zero the parameter gradients
        optim.zero_grad()

        # forward + backward + optimize
        sample = model(X_mb)
        loss = mse_loss(X_mb_orig, sample)
        loss.backward()
        optim.step()
        running_loss += loss.item()

        logging.debug("Iteration {} loss: {:.4}".format(it, running_loss))

    # save the model
    np.savetxt(model_dir + "AE_enc_W1", model.enc[0].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b1", model.enc[0].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W2", model.enc[2].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b2", model.enc[2].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W3", model.enc[4].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b3", model.enc[4].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W4", model.enc[6].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b4", model.enc[6].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W5", model.enc[8].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b5", model.enc[8].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_W6", model.enc[10].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_enc_b6", model.enc[10].bias.data, delimiter=",")

    np.savetxt(model_dir + "AE_dec_W1", model.dec[0].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b1", model.dec[0].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W2", model.dec[2].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b2", model.dec[2].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W3", model.dec[4].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b3", model.dec[4].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W4", model.dec[6].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b4", model.dec[6].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W5", model.dec[8].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b5", model.dec[8].bias.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_W6", model.dec[10].weight.data, delimiter=",")
    np.savetxt(model_dir + "AE_dec_b6", model.dec[10].bias.data, delimiter=",")


def build_model(model_dir, normalization_params_dir):
    AE_enc_W1 = np.loadtxt(model_dir + "AE_enc_W1", delimiter=",")
    AE_enc_b1 = np.loadtxt(model_dir + "AE_enc_b1", delimiter=",")
    AE_enc_W2 = np.loadtxt(model_dir + "AE_enc_W2", delimiter=",")
    AE_enc_b2 = np.loadtxt(model_dir + "AE_enc_b2", delimiter=",")
    AE_enc_W3 = np.loadtxt(model_dir + "AE_enc_W3", delimiter=",")
    AE_enc_b3 = np.loadtxt(model_dir + "AE_enc_b3", delimiter=",")
    AE_enc_W4 = np.loadtxt(model_dir + "AE_enc_W4", delimiter=",")
    AE_enc_b4 = np.loadtxt(model_dir + "AE_enc_b4", delimiter=",")
    AE_enc_W5 = np.loadtxt(model_dir + "AE_enc_W5", delimiter=",")
    AE_enc_b5 = np.loadtxt(model_dir + "AE_enc_b5", delimiter=",")
    AE_enc_W6 = np.loadtxt(model_dir + "AE_enc_W6", delimiter=",")
    AE_enc_b6 = np.loadtxt(model_dir + "AE_enc_b6", delimiter=",")

    AE_dec_W1 = np.loadtxt(model_dir + "AE_dec_W1", delimiter=",")
    AE_dec_b1 = np.loadtxt(model_dir + "AE_dec_b1", delimiter=",")
    AE_dec_W2 = np.loadtxt(model_dir + "AE_dec_W2", delimiter=",")
    AE_dec_b2 = np.loadtxt(model_dir + "AE_dec_b2", delimiter=",")
    AE_dec_W3 = np.loadtxt(model_dir + "AE_dec_W3", delimiter=",")
    AE_dec_b3 = np.loadtxt(model_dir + "AE_dec_b3", delimiter=",")
    AE_dec_W4 = np.loadtxt(model_dir + "AE_dec_W4", delimiter=",")
    AE_dec_b4 = np.loadtxt(model_dir + "AE_dec_b4", delimiter=",")
    AE_dec_W5 = np.loadtxt(model_dir + "AE_dec_W5", delimiter=",")
    AE_dec_b5 = np.loadtxt(model_dir + "AE_dec_b5", delimiter=",")
    AE_dec_W6 = np.loadtxt(model_dir + "AE_dec_W6", delimiter=",")
    AE_dec_b6 = np.loadtxt(model_dir + "AE_dec_b6", delimiter=",")

    _, dim_net = AE_enc_W1.shape
    print(dim_net)
    model = AE(dim_net)
    with torch.no_grad():
        model.enc[0].weight.copy_(torch.from_numpy(AE_enc_W1).float())
        model.enc[0].bias.copy_(torch.from_numpy(AE_enc_b1).float())
        model.enc[2].weight.copy_(torch.from_numpy(AE_enc_W2).float())
        model.enc[2].bias.copy_(torch.from_numpy(AE_enc_b2).float())
        model.enc[4].weight.copy_(torch.from_numpy(AE_enc_W3).float())
        model.enc[4].bias.copy_(torch.from_numpy(AE_enc_b3).float())
        model.enc[6].weight.copy_(torch.from_numpy(AE_enc_W4).float())
        model.enc[6].bias.copy_(torch.from_numpy(AE_enc_b4).float())
        model.enc[8].weight.copy_(torch.from_numpy(AE_enc_W5).float())
        model.enc[8].bias.copy_(torch.from_numpy(AE_enc_b5).float())
        model.enc[10].weight.copy_(torch.from_numpy(AE_enc_W6).float())
        model.enc[10].bias.copy_(torch.from_numpy(AE_enc_b6).float())

        model.dec[0].weight.copy_(torch.from_numpy(AE_dec_W1).float())
        model.dec[0].bias.copy_(torch.from_numpy(AE_dec_b1).float())
        model.dec[2].weight.copy_(torch.from_numpy(AE_dec_W2).float())
        model.dec[2].bias.copy_(torch.from_numpy(AE_dec_b2).float())
        model.dec[4].weight.copy_(torch.from_numpy(AE_dec_W3).float())
        model.dec[4].bias.copy_(torch.from_numpy(AE_dec_b3).float())
        model.dec[6].weight.copy_(torch.from_numpy(AE_dec_W4).float())
        model.dec[6].bias.copy_(torch.from_numpy(AE_dec_b4).float())
        model.dec[8].weight.copy_(torch.from_numpy(AE_dec_W5).float())
        model.dec[8].bias.copy_(torch.from_numpy(AE_dec_b5).float())
        model.dec[10].weight.copy_(torch.from_numpy(AE_dec_W6).float())
        model.dec[10].bias.copy_(torch.from_numpy(AE_dec_b6).float())

    norm_params_in = {"min_val": [], "max_val": []}
    norm_params_in["min_val"] = np.loadtxt(normalization_params_dir + "min_val")
    norm_params_in["max_val"] = np.loadtxt(normalization_params_dir + "max_val")

    return model, norm_params_in


def normalize(data, norm_params):
    _, dim = data.shape
    norm_data = data.copy()

    min_val = norm_params["min_val"]
    max_val = norm_params["max_val"]

    # For each dimension
    for i in range(dim):
        norm_data[:, i] = (norm_data[:, i] - min_val[i]) / (max_val[i] - min_val[i] + 1e-6)

    return norm_data


def train_loss(model, model_parameters, loss_threshold=0.113, skip_r=100):
    orig_features = 52
    logging.basicConfig(filename="./debug/train_loss.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    no_lags = int(model_parameters["lags"])
    norm_params_in = model_parameters["norm_parameters"]

    mse_loss = nn.MSELoss(reduction="mean")

    model.eval()
    loss_dist = []
    from glob import glob

    # fnames = glob('data/fault_free_training/fault_free_training_p1/*')
    fnames = glob("data/faulty_training/faulty_training_4/faulty_training_4_p1/fault*")
    number_of_samples = 0
    number_of_false_alarms = 0
    for f in fnames:
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r)]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
        norm_data_x = normalize(data_x, norm_params_in)
        X_mb = norm_data_x

        # create tensors
        X_mb = torch.tensor(X_mb).float()

        # forward
        for data_i in X_mb:
            sample = model(data_i)
            loss = mse_loss(data_i[:, :orig_features], sample)
            running_loss = loss.item()
            number_of_samples += 1
            if running_loss < loss_threshold:
                number_of_false_alarms += 1
                loss_dist.append(running_loss)

                # logging.debug('Iteration loss: {:.4}'.format(running_loss))
            # logging.debug('Iteration loss: {:.4}'.format(running_loss))
        logging.debug("finished file: **************")
        logging.debug("Ratio: {:.4}".format(float(number_of_false_alarms) / number_of_samples))
        logging.debug(str(f))

    logging.debug("Number of false alarms: {}".format(number_of_false_alarms))
    logging.debug("Number of samples: {}".format(number_of_samples))
    logging.debug("Ratio: {:.4}".format(float(number_of_false_alarms) / number_of_samples))


def choose_loss_threshold(loss_arr, percentage_of_data_withing_bound):
    # loss_arr = loss_arr.sort()
    len_loss_arr = len(loss_arr)
    threshold_temp = np.mean(loss_arr)
    while float(sum(map(lambda x: x < threshold_temp, loss_arr))) / len_loss_arr < percentage_of_data_withing_bound:
        threshold_temp += 0.001
    return threshold_temp


def train_loss_fast(model, model_parameters, perc_of_data_withing_bound=0.95, skip_r=100):
    orig_features = 52
    logging.basicConfig(filename="./debug/train_loss.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    no_lags = int(model_parameters["lags"])
    norm_params_in = model_parameters["norm_parameters"]

    # mse_loss = nn.MSELoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="none")

    model.eval()
    loss_dist = []
    # TODO this is where i stopped
    from glob import glob

    fnames = glob("data/fault_free_training/fault_free_training_p2/*")
    # fnames = glob('data/faulty_training/faulty_training_4/faulty_training_4_p1/fault*')
    number_of_samples = 0
    number_of_false_alarms = 0
    loss_threshold = []
    for f in fnames:
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r)]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
        norm_data_x = normalize(data_x, norm_params_in)
        X_mb = norm_data_x

        # create tensors
        X_mb = torch.tensor(X_mb).float()

        # forward
        sample = model(X_mb)
        loss = mse_loss(X_mb[:, :orig_features], sample)
        loss_per_sample = np.mean(loss.detach().numpy(), axis=1)
        loss_threshold.append(choose_loss_threshold(loss_per_sample, perc_of_data_withing_bound))
        # if loss_threshold_temp > loss_threshold:
        #     loss_threshold = loss_threshold_temp
        false_alarm_ratio = float(sum(map(lambda x: x < loss_threshold[-1], loss_per_sample))) / len(loss_per_sample)
        logging.debug("false alarm ratio: {:.6}".format(false_alarm_ratio))
        logging.debug("loss_threshold is: {:.6}".format(loss_threshold[-1]))
        logging.debug("finished file: **************")
        logging.debug(str(f))
    threshold_mean_e = np.mean(loss_threshold)
    threshold_len_e = len(loss_threshold)
    loss_threshold.sort()
    threshold_idx = int(perc_of_data_withing_bound * threshold_len_e)
    logging.debug("the mean threshold is: {:.6}".format(threshold_mean_e))
    logging.debug("chosen threshold: {:.6}".format(loss_threshold[threshold_idx]))


def detect_imputed_anomalies(
    model,
    model_parameters,
    faulty_sensor,
    fault_no,
    loss_threshold=0.113,
    data_set="gan",
    skip_r=160,
):
    orig_features = 52
    logging.basicConfig(filename="./debug/anomaly_detect.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    no_lags = int(model_parameters["lags"])
    norm_params_in = model_parameters["norm_parameters"]

    # mse_loss = nn.MSELoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="none")

    model.eval()
    loss_dist = []
    # TODO this is where i stopped
    from glob import glob

    if data_set == "gan":
        if fault_no == 0:
            fnames = glob("data_imputed/fault_free/*/*")
        else:
            # fnames = glob('data/faulty_training/faulty_training_*/faulty_training_*_p1/fault*')
            fnames = glob("data_imputed/faulty/" + str(faulty_sensor) + "/*")
    elif data_set == "stupid":
        if fault_no == 0:
            fnames = glob("data_stupid/fault_free/*/*")
        else:
            # fnames = glob('data/faulty_training/faulty_training_*/faulty_training_*_p1/fault*')
            fnames = glob("data_stupid/faulty/" + str(faulty_sensor) + "/*")
    number_of_samples = 0
    number_of_false_alarms = 0
    false_alarm_ratio_arr = []
    loss_mean_arr = []
    for f in fnames:
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r)]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
        norm_data_x = normalize(data_x, norm_params_in)
        X_mb = norm_data_x

        # create tensors
        X_mb = torch.tensor(X_mb).float()

        # forward
        sample = model(X_mb)
        loss = mse_loss(X_mb[:, :orig_features], sample)
        # running_loss = loss.item()
        # number_of_samples += 1
        # if running_loss < loss_threshold:
        #     number_of_false_alarms += 1
        #     loss_dist.append(running_loss)

        # logging.debug('file loss:')
        # logging.debug(loss)
        # logging.debug(loss.shape)
        loss_mean = np.mean(loss.detach().numpy(), axis=1)
        loss_mean_arr.append(loss_mean)
        logging.debug(loss_mean)
        # logging.debug('numpy loss:')
        # logging.debug('loss_mean len: {}'.format(len(loss_mean)))
        # compute the ratio of detected anomalies
        false_alarm_ratio = float(sum(map(lambda x: x < loss_threshold, loss_mean))) / len(loss_mean)
        false_alarm_ratio_arr.append(false_alarm_ratio)
        logging.debug("false alarm ratio: {:.6}".format(false_alarm_ratio))
        # logging.debug('Iteration loss: {:.4}'.format(running_loss))
        # logging.debug('Iteration loss: {:.4}'.format(running_loss))
        logging.debug("finished file: **************")
        # logging.debug('Ratio: {:.4}'.format(float(number_of_false_alarms)/number_of_samples))
        logging.debug(str(f))
    overall_false_alarm_ratio = np.mean(false_alarm_ratio_arr)
    with open("./results/false_alarm_rate_" + str(fault_no) + ".txt", "w") as f:
        for arr_i in loss_mean_arr:
            for element in arr_i:
                f.write(str(element))
                f.write(",")
            f.write("\n")

    # loss_mean_arr_str = map(str,loss_mean_arr)
    # f.write('\n'.join(loss_mean_arr_str))
    logging.debug("false alarm ratio overall: {:.6}".format(overall_false_alarm_ratio))
    return overall_false_alarm_ratio


def detect_anomalies(model, model_parameters, fault_no, loss_threshold=0.113, skip_r=160):
    orig_features = 52
    logging.basicConfig(filename="./debug/anomaly_detect.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    no_lags = int(model_parameters["lags"])
    norm_params_in = model_parameters["norm_parameters"]

    # mse_loss = nn.MSELoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="none")

    model.eval()
    loss_dist = []
    # TODO this is where i stopped
    from glob import glob

    if fault_no == 0:
        fnames = glob("data/fault_free_training/fault_free_training_p2/*")
    else:
        # fnames = glob('data/faulty_training/faulty_training_*/faulty_training_*_p1/fault*')
        fnames = glob(
            "data/faulty_training/faulty_training_" + str(fault_no) + "/faulty_training_" + str(fault_no) + "_p1/fault*"
        )
    number_of_samples = 0
    number_of_false_alarms = 0
    false_alarm_ratio_arr = []
    loss_mean_arr = []
    for f in fnames:
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r)]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
        norm_data_x = normalize(data_x, norm_params_in)
        X_mb = norm_data_x

        # create tensors
        X_mb = torch.tensor(X_mb).float()

        # forward
        sample = model(X_mb)
        loss = mse_loss(X_mb[:, :orig_features], sample)
        # running_loss = loss.item()
        # number_of_samples += 1
        # if running_loss < loss_threshold:
        #     number_of_false_alarms += 1
        #     loss_dist.append(running_loss)

        # logging.debug('file loss:')
        # logging.debug(loss)
        # logging.debug(loss.shape)
        loss_mean = np.mean(loss.detach().numpy(), axis=1)
        loss_mean_arr.append(loss_mean)
        logging.debug(loss_mean)
        # logging.debug('numpy loss:')
        # logging.debug('loss_mean len: {}'.format(len(loss_mean)))
        # compute the ratio of detected anomalies
        false_alarm_ratio = float(sum(map(lambda x: x < loss_threshold, loss_mean))) / len(loss_mean)
        false_alarm_ratio_arr.append(false_alarm_ratio)
        logging.debug("false alarm ratio: {:.6}".format(false_alarm_ratio))
        # logging.debug('Iteration loss: {:.4}'.format(running_loss))
        # logging.debug('Iteration loss: {:.4}'.format(running_loss))
        logging.debug("finished file: **************")
        # logging.debug('Ratio: {:.4}'.format(float(number_of_false_alarms)/number_of_samples))
        logging.debug(str(f))
    overall_false_alarm_ratio = np.mean(false_alarm_ratio_arr)
    with open("./results/false_alarm_rate_" + str(fault_no) + ".txt", "w") as f:
        for arr_i in loss_mean_arr:
            for element in arr_i:
                f.write(str(element))
                f.write(",")
            f.write("\n")

    # loss_mean_arr_str = map(str,loss_mean_arr)
    # f.write('\n'.join(loss_mean_arr_str))
    logging.debug("false alarm ratio overall: {:.6}".format(overall_false_alarm_ratio))
    return overall_false_alarm_ratio

    # logging.debug('Number of false alarms: {}'.format(number_of_false_alarms))
    # logging.debug('Number of samples: {}'.format(number_of_samples))
    # logging.debug('Ratio: {:.4}'.format(float(number_of_false_alarms)/number_of_samples))


def detect_anomalies2(model, model_parameters, fault_no, loss_threshold_arr, perc_of_outputs, skip_r=100):
    orig_features = 52
    logging.basicConfig(filename="./debug/anomaly_detect.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    no_lags = int(model_parameters["lags"])
    norm_params_in = model_parameters["norm_parameters"]

    # mse_loss = nn.MSELoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="none")

    model.eval()
    loss_dist = []
    # TODO this is where i stopped
    from glob import glob

    if fault_no == 0:
        fnames = glob("data/fault_free_training/fault_free_training_p2/*")
    else:
        # fnames = glob('data/faulty_training/faulty_training_*/faulty_training_*_p1/fault*')
        fnames = glob(
            "data/faulty_training/faulty_training_" + str(fault_no) + "/faulty_training_" + str(fault_no) + "_p1/fault*"
        )
    number_of_samples = 0
    number_of_false_alarms = 0
    false_alarm_ratio_arr = []
    loss_mean_arr = []
    for f in fnames:
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r)]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
        norm_data_x = normalize(data_x, norm_params_in)
        X_mb = norm_data_x

        # create tensors
        X_mb = torch.tensor(X_mb).float()

        # forward
        sample = model(X_mb)
        loss = mse_loss(X_mb[:, :orig_features], sample)
        # running_loss = loss.item()
        # number_of_samples += 1
        # if running_loss < loss_threshold:
        #     number_of_false_alarms += 1
        #     loss_dist.append(running_loss)

        # logging.debug('file loss:')
        # logging.debug(loss)
        # logging.debug(loss.shape)
        # loss_mean = np.mean(loss.detach().numpy(), axis=1)
        loss_mean = loss.detach().numpy()
        # loss_mean = np.sqrt(loss_mean)
        for loss_mean_i in loss_mean:
            compare_loss = [l_m < th_i for l_m, th_i in zip(loss_mean_i, loss_threshold_arr)]
            if np.mean(compare_loss) > perc_of_outputs:
                false_alarm_ratio_arr.append(1)
            else:
                false_alarm_ratio_arr.append(0)
        # loss_mean_arr.append(loss_mean)
        # logging.debug(loss_mean)
        # logging.debug(len(loss_mean[0]))
        logging.debug("*********************")
        logging.debug(np.mean(false_alarm_ratio_arr))
        logging.debug("*********************")
        return np.mean(false_alarm_ratio_arr)
        # logging.debug('numpy loss:')
        # logging.debug('loss_mean len: {}'.format(len(loss_mean)))
        # compute the ratio of detected anomalies
    #     false_alarm_ratio = float(sum(map(lambda x: x < loss_threshold, loss_mean)))/len(loss_mean)
    #     false_alarm_ratio_arr.append(false_alarm_ratio)
    #     logging.debug('false alarm ratio: {:.6}'.format(false_alarm_ratio))
    #     # logging.debug('Iteration loss: {:.4}'.format(running_loss))
    #     # logging.debug('Iteration loss: {:.4}'.format(running_loss))
    #     logging.debug('finished file: **************')
    #     # logging.debug('Ratio: {:.4}'.format(float(number_of_false_alarms)/number_of_samples))
    #     logging.debug(str(f))
    # overall_false_alarm_ratio = np.mean(false_alarm_ratio_arr)
    # with open('./results/false_alarm_rate_'+str(fault_no)+'.txt', 'w') as f:
    #     for arr_i in loss_mean_arr:
    #         for element in arr_i:
    #             f.write(str(element))
    #             f.write(',')
    #         f.write('\n')

    # # loss_mean_arr_str = map(str,loss_mean_arr)
    # # f.write('\n'.join(loss_mean_arr_str))
    # logging.debug('false alarm ratio overall: {:.6}'.format(overall_false_alarm_ratio))
    # return overall_false_alarm_ratio

    # # logging.debug('Number of false alarms: {}'.format(number_of_false_alarms))
    # # logging.debug('Number of samples: {}'.format(number_of_samples))
    # # logging.debug('Ratio: {:.4}'.format(float(number_of_false_alarms)/number_of_samples))


def train_loss_fast2(model, model_parameters, perc_of_data_withing_bound=0.95, skip_r=100):
    orig_features = 52
    logging.basicConfig(filename="./debug/train_loss.log", level=logging.DEBUG)

    mb_size = int(model_parameters["batch_size"])
    no_lags = int(model_parameters["lags"])
    norm_params_in = model_parameters["norm_parameters"]

    # mse_loss = nn.MSELoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="none")
    # mse_loss = nn.MSELoss(reduction="mean")

    model.eval()
    loss_dist = []
    # TODO this is where i stopped
    from glob import glob

    fnames = glob("data/fault_free_training/fault_free_training_p2/*")
    # fnames = glob('data/faulty_training/faulty_training_4/faulty_training_4_p1/fault*')
    number_of_samples = 0
    number_of_false_alarms = 0
    loss_threshold = []
    loss_dict_overall = {i: [] for i in range(orig_features)}
    for f in fnames:
        data_arr = [np.loadtxt(f, delimiter=",", skiprows=skip_r)]
        data_arr = introduce_delayed_measurements(data_arr, no_lags)
        data_x = np.concatenate(data_arr)
        norm_data_x = normalize(data_x, norm_params_in)
        X_mb = norm_data_x

        # create tensors
        X_mb = torch.tensor(X_mb).float()

        # forward
        sample = model(X_mb)
        loss = mse_loss(X_mb[:, :orig_features], sample)
        loss_per_sample = loss.detach().numpy()
        loss_dict = {i_d: [j_d for j_d in loss_per_sample[i_d]] for i_d in range(len(loss_per_sample))}
        # logging.debug(loss_per_sample)
        # logging.debug(len(loss_per_sample))
        # logging.debug(len(loss_per_sample[0]))
        # logging.debug(loss_dict)
        # logging.debug("******************")
        for key_i in loss_dict_overall.keys():
            loss_dict_overall[key_i] += loss_dict[key_i]

    # logging.debug("******************")
    # logging.debug(loss_dict_overall)
    # logging.debug(len(loss_dict_overall[0]))
    # logging.debug("******************")
    loss_threshold_arr = [0 for _ in range(orig_features)]
    threshold_idx = int(perc_of_data_withing_bound * len(loss_dict_overall[0]))
    for key_i in loss_dict_overall.keys():
        loss_dict_overall[key_i].sort()
        loss_threshold_arr[key_i] = loss_dict_overall[key_i][threshold_idx]
    # logging.debug(loss_threshold_arr)
    return loss_threshold_arr

    #     loss_threshold.append(choose_loss_threshold(loss_per_sample, perc_of_data_withing_bound))
    #     # if loss_threshold_temp > loss_threshold:
    #     #     loss_threshold = loss_threshold_temp
    #     false_alarm_ratio = float(sum(map(lambda x: x < loss_threshold[-1], loss_per_sample)))/len(loss_per_sample)
    #     logging.debug('false alarm ratio: {:.6}'.format(false_alarm_ratio))
    #     logging.debug('loss_threshold is: {:.6}'.format(loss_threshold[-1]))
    #     logging.debug('finished file: **************')
    #     logging.debug(str(f))
    # threshold_mean_e = np.mean(loss_threshold)
    # threshold_len_e = len(loss_threshold)
    # loss_threshold.sort()
    # threshold_idx = int(perc_of_data_withing_bound*threshold_len_e)
    # logging.debug('the mean threshold is: {:.6}'.format(threshold_mean_e))
    # logging.debug('chosen threshold: {:.6}'.format(loss_threshold[threshold_idx]))
