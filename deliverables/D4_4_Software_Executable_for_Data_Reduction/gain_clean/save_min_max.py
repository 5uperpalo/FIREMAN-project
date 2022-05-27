import argparse
import logging
import os

import numpy as np
import torch
from data_loader import data_loader

from utils import normalization, renormalization, rounding


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
    mb_size = 512
    # 2. Missing rate
    p_miss = 0.1
    # 3. Hint rate
    p_hint = 0.9
    # 4. Loss Hyperparameters
    alpha = 100
    # 6. No

    # define mask matrix
    data_m = 1 - np.isnan(data_x)
    no, dim = data_x.shape
    h_dim = int(dim)

    # trainX = np.loadtxt(data_path, delimiter=",", skiprows=1)
    # no, dim = trainX.shape
    # trainM = sample_M(no, dim, p_miss)

    # todo Normalization
    norm_data, norm_parameters = normalization(data_x)

    folder_name = "normalization_params/"
    np.savetxt(folder_name + "min_val", norm_parameters["min_val"], delimiter=",")
    np.savetxt(folder_name + "max_val", norm_parameters["max_val"], delimiter=",")


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

    gain_parameters = {
        "batch_size": args.batch_size,
        "hint_rate": args.hint_rate,
        "alpha": args.alpha,
        "folder_name": args.folder_name,
        "iterations": args.iterations,
    }

    print("calling data loader \n")
    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
    print("finished calling data loader \n")
    print("calling gain function \n")

    # Impute missing data
    train_gain(miss_data_x, gain_parameters)


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
    parser.add_argument("--hint_rate", help="hint probability", default=0.9, type=float)
    parser.add_argument("--alpha", help="hyperparameter", default=100, type=float)
    parser.add_argument("--iterations", help="number of training interations", default=10000, type=int)
    parser.add_argument("--folder_name", default="0", type=str)
    args = parser.parse_args()

    # Calls main function
    # imputed_data, rmse = main(args)
    main(args)
