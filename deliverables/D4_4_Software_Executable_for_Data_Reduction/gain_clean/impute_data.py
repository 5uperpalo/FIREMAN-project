"""GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
"""

import logging
import os

import numpy as np

# Necessary packages
import tensorflow as tf
from tqdm import tqdm

from utils import (
    binary_sampler,
    normalization,
    renormalization,
    rounding,
    sample_batch_index,
    uniform_sampler,
    xavier_init,
)


def impute_data(fnames, gain_parameters):
    impute_logging_file = "debug/" + str(gain_parameters["data_path"]) + ".log"
    logging.basicConfig(filename=impute_logging_file, level=logging.DEBUG)

    batch_size = int(gain_parameters["batch_size"])
    miss_rate = float(gain_parameters["miss_rate"])

    # System parameters
    alpha = 100  # gain_parameters['alpha']

    # Other parameters
    dim = 52

    # Hidden state dimensions
    h_dim = int(dim)

    # # # logging.debug(str(h_dim))
    # logging.debug(str(dim))
    # logging.debug("***********************************************")

    ## GAIN architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN functions
    # Generator
    def generator(x, m):
        # logging.debug('generator called')
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # logging.debug('discriminator called')
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## GAIN structure
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1.0 - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    sess_2 = tf.Session()
    saver_2 = tf.train.Saver()
    model_folder = "./model/" + str(batch_size) + "/my-test-model"
    saver_2.restore(sess_2, model_folder)

    # logging.debug('start imputing the data')
    ## Return imputed data

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
        # ori_data_x = data_x

        # Normalization before change
        # norm_data, norm_parameters = normalization(data_x)
        # norm_data_x = np.nan_to_num(norm_data, 0)
        # after change
        norm_data, norm_parameters = normalization(miss_data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # Define mask matrix
        # data_m = 1-np.isnan(data_x)
        # Other parameters
        # no, dim = data_x.shape
        Z_mb = uniform_sampler(0, 0.01, no, dim)
        M_mb = data_m
        X_mb = norm_data_x
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        # logging.debug('start sess.run for data imputation')
        imputed_data = sess_2.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

        # logging.debug('finished sess.run for data imputation')
        imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

        # logging.debug('start renormalization')
        # Renormalization
        imputed_data = renormalization(imputed_data, norm_parameters)

        # logging.debug('start rounding')
        # Rounding
        # imputed_data = rounding(imputed_data, data_x)
        # logging.debug('finished roudning')

        # Impute missing data
        # imputed_data_x = gain(miss_data_x, gain_parameters)
        # imputed_data_x = impute_data(miss_data_x, batch_size)
        out_file_name = (
            "./results/" + str(batch_size) + "_" + str(gain_parameters["data_name"]) + "/" + str(os.path.basename(f))
        )
        with open(out_file_name, "w") as f:
            np.savetxt(f, imputed_data, delimiter=",")
        # np.savetxt(out_file_name, imputed_data, delimiter=",")

    # np.savetxt("original_data.csv", norm_data_x[batch_idx,:], delimiter=",")
    # np.savetxt("imputed_data.csv", imputed_data, delimiter=",")

    return imputed_data
