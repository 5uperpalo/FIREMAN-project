import argparse
import logging
import os

import numpy as np
import torch
from data_loader import data_loader

from utils import binary_sampler_with_lags, normalization, renormalization, rounding


def persistent_sensor_failure_mask(rows, cols, failing_sensor_ids):
    mask_matrix = np.ones((rows, cols))
    mask_matrix[:, failing_sensor_ids] = 0
    return mask_matrix


def nan_to_avg_of_feature(data_x, norm_parameters):
    norm_data = data_x.copy()
    avg_of_features = np.loadtxt("./data/mean_values_faultfree", delimiter=",")
    avg_of_features = avg_of_features.reshape(1, avg_of_features.shape[0])
    min_val = norm_parameters["min_val"]
    min_val = min_val.reshape(1, min_val.shape[0])
    min_val = min_val[:, : avg_of_features.shape[1]]
    max_val = norm_parameters["max_val"]
    max_val = max_val.reshape(1, max_val.shape[0])
    max_val = max_val[:, : avg_of_features.shape[1]]
    avg_of_features = (avg_of_features - min_val) / (max_val + 1e-6)
    # now replace the nan with the average value
    inds = np.where(np.isnan(norm_data))
    norm_data[inds] = np.take(avg_of_features, inds[1])
    return norm_data


# Mask Vector and Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0.0, 1.0, size=[m, n])
    B = A > p
    C = 1.0 * B
    return C


#%% 3. Others
# Random sample generator for Z
def sample_Z(m, n):
    # return np.random.uniform(0., 1., size = [m, n])
    return np.random.uniform(0.0, 0.01, size=[m, n])


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


class NetD(torch.nn.Module):
    def __init__(self, dim, h_dim, orig_features):
        super(NetD, self).__init__()
        self.fc1 = torch.nn.Linear(dim + orig_features, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, h_dim)
        # self.fc2 = torch.nn.Linear(h_dim, 52)
        # self.fc_additional1 = torch.nn.Linear(52, h_dim)
        self.fc3 = torch.nn.Linear(h_dim, orig_features)
        # self.fc3 = torch.nn.Linear(52, dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()
        self.orig_features = orig_features

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        # layers = [self.fc1, self.fc2, self.fc_additional1, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, x, m, g, h):
        """Eq(3)"""
        inp = m * x + (1 - m) * g
        inp = torch.cat((inp, h), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        # out = self.relu(self.fc_additional1(out))
        out = self.sigmoid(self.fc3(out))  # [0,1] Probability Output
        # out = self.fc3(out)

        return out


"""
Eq(2)
"""


class NetG(torch.nn.Module):
    def __init__(self, dim, h_dim, orig_features):
        super(NetG, self).__init__()
        self.fc1 = torch.nn.Linear(dim + orig_features, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, h_dim)
        # this layer was added to increase the size of the nn after adding lags of the dataset
        # self.fc_lag1 = torch.nn.Linear(h_dim, h_dim)
        self.fc3 = torch.nn.Linear(h_dim, orig_features)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        # layers = [self.fc1, self.fc2, self.fc_lag1, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    # def forward(self, x, z, m):
    def forward(self, x, m):
        # inp = m * x + (1-m) * z
        # inp = torch.cat((inp, m), dim=1)
        inp = torch.cat((x, m), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        # out = self.relu(self.fc_lag1(out))
        out = self.sigmoid(self.fc3(out))  # [0,1] Probability Output
        #         out = self.fc3(out)

        return out


def train_gain(data_x, gain_parameters):
    orig_features = 52
    logging.basicConfig(filename="./debug/train_gain.log", level=logging.DEBUG)

    # 1. Mini batch size
    # mb_size = 128
    # mb_size = 512
    mb_size = 1024
    # 2. Missing rate
    p_miss = 0.1
    # 3. Hint rate
    # p_hint = 0.9
    p_hint = 0.5
    # 4. Loss Hyperparameters
    alpha = 100
    # 6. No

    # define mask matrix
    # data_m = 1-np.isnan(data_x)
    no, dim = data_x.shape
    h_dim = int(dim)

    netD = NetD(dim, h_dim, orig_features)
    # netD = NetD(orig_features, orig_features, orig_features)
    netG = NetG(dim, h_dim, orig_features)

    optimD = torch.optim.Adam(netD.parameters(), lr=0.001)
    optimG = torch.optim.Adam(netG.parameters(), lr=0.001)

    # bce_loss = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
    # mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    mse_loss = torch.nn.MSELoss(reduction="mean")

    i = 1
    miss_rate = float(gain_parameters["miss_rate"])
    #%% Start Iterations
    for it in range(1000):
        logging.debug(it)
        #%% Inputs
        mb_idx = sample_idx(no, mb_size)
        # X_mb = norm_data_x[mb_idx,:]
        X_mb = data_x[mb_idx, :]
        # Introduce missing data
        data_m = binary_sampler_with_lags(1 - miss_rate, mb_size, dim, orig_features)
        # using persistent mask (persistent sensor failure)
        # failing_sensor_ids = 51
        # data_m = persistent_sensor_failure_mask(mb_size, dim, failing_sensor_ids)
        X_mb[data_m == 0] = 0
        X_mb_orig = X_mb[:, :orig_features]

        Z_mb = sample_Z(mb_size, dim)

        # M_mb = data_m[mb_idx,:]
        M_mb = data_m
        M_mb_orig = M_mb[:, :orig_features]
        H_mb1 = sample_M(mb_size, orig_features, 1 - p_hint)
        H_mb = M_mb_orig * H_mb1 + 0.5 * (1 - H_mb1)
        # H_mb = M_mb_orig

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        X_mb = torch.tensor(X_mb).float()
        X_mb_orig = torch.tensor(X_mb_orig).float()
        New_X_mb = torch.tensor(New_X_mb).float()
        Z_mb = torch.tensor(Z_mb).float()
        M_mb = torch.tensor(M_mb).float()
        M_mb_orig = torch.tensor(M_mb_orig).float()
        H_mb = torch.tensor(H_mb).float()
        H_mb1 = torch.tensor(H_mb1).float()

        # Train D
        G_sample = netG(New_X_mb, M_mb_orig)
        # D_prob = netD(X_mb_orig, M_mb_orig, G_sample, H_mb)
        G_sample = torch.cat((G_sample, X_mb[:, orig_features:]), dim=1)
        D_prob = netD(X_mb, M_mb, G_sample, H_mb)
        # D_loss = bce_loss(D_prob, M_mb_orig)
        D_loss = bce_loss((1 - H_mb1) * M_mb_orig, (1 - H_mb1) * D_prob)
        # D_loss = bce_loss(M_mb_orig, D_prob)

        D_loss.backward()
        optimD.step()
        optimD.zero_grad()

        # Train G
        G_sample = netG(New_X_mb, M_mb_orig)
        G_sample1 = torch.cat((G_sample, X_mb[:, orig_features:]), dim=1)
        D_prob = netD(X_mb, M_mb, G_sample1, H_mb)
        D_prob.detach_()
        # G_loss1 = ((1 - M_mb_orig) * (torch.sigmoid(D_prob)+1e-8).log()).mean()/(1-M_mb_orig).sum()
        # G_mse_loss = mse_loss(M_mb_orig*X_mb_orig, M_mb_orig*G_sample) / M_mb_orig.sum()
        # G_loss1 = ((1 - M_mb_orig) * (torch.sigmoid(D_prob)+1e-8).log()).mean()
        # G_mse_loss = mse_loss(M_mb_orig*X_mb_orig, M_mb_orig*G_sample)
        G_loss1 = -((1 - H_mb1) * ((1 - M_mb_orig) * ((D_prob) + 1e-8).log())).sum() / (1 - H_mb1).sum()
        # G_loss1 = -((1 - M_mb_orig) * ((D_prob)+1e-8).log()).sum()/(1 - M_mb_orig).sum()
        G_mse_loss = mse_loss(M_mb_orig * G_sample, M_mb_orig * X_mb_orig)
        G_loss = G_loss1 + alpha * G_mse_loss

        G_loss.backward()
        optimG.step()
        optimG.zero_grad()

        logging.debug(D_loss)
        logging.debug(G_loss)
        logging.debug(G_loss1)
        logging.debug(G_mse_loss)
        logging.debug("********************")

        #%% Intermediate Losses
        if it % 100 == 0:
            print("Iter: {}".format(it))
            print("D_loss: {:.4}".format(D_loss))
            print("Train_loss: {:.4}".format(G_mse_loss))
            print()

    # folder_name = 'weights/' + str(gain_parameters['folder_name'])
    folder_name = "weights/" + "alpha_" + str(alpha) + "_p" + str(gain_parameters["folder_name"]) + "/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.savetxt(folder_name + "G_W1", netG.fc1.weight.data, delimiter=",")
    np.savetxt(folder_name + "G_b1", netG.fc1.bias.data, delimiter=",")
    np.savetxt(folder_name + "G_W2", netG.fc2.weight.data, delimiter=",")
    np.savetxt(folder_name + "G_b2", netG.fc2.bias.data, delimiter=",")
    # np.savetxt(folder_name+'weights/G_W3', netG.fc_lag1.weight.data, delimiter=',')
    # np.savetxt(folder_name+'weights/G_b3', netG.fc_lag1.bias.data, delimiter=',')
    np.savetxt(folder_name + "G_W4", netG.fc3.weight.data, delimiter=",")
    np.savetxt(folder_name + "G_b4", netG.fc3.bias.data, delimiter=",")
    # np.savetxt(folder_name+'weights/G_W3', netG.fc3.weight.data, delimiter=',')
    # np.savetxt(folder_name+'weights/G_b3', netG.fc3.bias.data, delimiter=',')


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
    skip_rows = args.skip_rows

    print("calling data loader \n")
    # Load data and introduce missingness
    # ori_data_x, miss_data_x, data_m, norm_parameters = data_loader(data_name, miss_rate)
    norm_data, norm_parameters = data_loader(data_name, miss_rate, skip_rows)
    print("finished calling data loader \n")
    print("calling gain function \n")

    gain_parameters = {
        "batch_size": args.batch_size,
        "hint_rate": args.hint_rate,
        "alpha": args.alpha,
        "folder_name": args.folder_name,
        "norm_parameters": norm_parameters,
        "miss_rate": miss_rate,
        "iterations": args.iterations,
    }

    # Impute missing data
    # train_gain(miss_data_x, gain_parameters)
    train_gain(norm_data, gain_parameters)


if __name__ == "__main__":

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        choices=[
            "letter",
            "spam",
            "tennesee_dat",
            "fault_free_training",
            "faulty_training",
            "faulty_training_test",
        ],
        default="spam",
        type=str,
    )
    parser.add_argument("--miss_rate", help="missing data probability", default=0.1, type=float)
    parser.add_argument(
        "--batch_size",
        help="the number of samples in mini-batch",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--skip_rows",
        help="the number of rows to be skipped from the input file",
        default=1,
        type=int,
    )
    parser.add_argument("--hint_rate", help="hint probability", default=0.6, type=float)
    parser.add_argument("--alpha", help="hyperparameter", default=1000, type=float)
    parser.add_argument("--iterations", help="number of training interations", default=10000, type=int)
    parser.add_argument("--folder_name", default="0", type=str)
    args = parser.parse_args()

    # Calls main function
    # imputed_data, rmse = main(args)
    main(args)
