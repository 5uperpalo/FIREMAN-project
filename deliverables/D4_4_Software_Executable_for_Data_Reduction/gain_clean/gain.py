"""GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
"""

import logging

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


def classification_accuracy(class_score, true_label):
    return np.mean(np.around(class_score) == np.around(true_label))


def gain(data_x, gain_parameters):
    """Impute missing values in data_x

    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations

    Returns:
      - imputed_data: imputed data
    """

    orig_features = 52
    logging.basicConfig(filename="./debug/gain.log", level=logging.DEBUG)
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)

    # System parameters
    batch_size = gain_parameters["batch_size"]
    hint_rate = gain_parameters["hint_rate"]
    alpha = gain_parameters["alpha"]
    iterations = gain_parameters["iterations"]

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions
    # h_dim = int(dim)
    # because of lags
    h_dim = int(dim)

    logging.debug(str(h_dim))
    logging.debug(str(dim))
    logging.debug("***********************************************")

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## GAIN architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # # Mask vector
    # M = tf.placeholder(tf.float32, shape = [None, dim])
    # # Hint vector
    # H = tf.placeholder(tf.float32, shape = [None, dim])
    # because of lags
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, orig_features])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, orig_features])

    # Discriminator variables
    # D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
    # because of lags
    D_W1 = tf.Variable(xavier_init([orig_features * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    # D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    # D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
    # because of lags
    D_W3 = tf.Variable(xavier_init([h_dim, orig_features]))
    D_b3 = tf.Variable(tf.zeros(shape=[orig_features]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    # G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))
    # because of lags
    G_W1 = tf.Variable(xavier_init([dim + orig_features, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    # G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    # G_b3 = tf.Variable(tf.zeros(shape = [dim]))
    # because of lags
    G_W3 = tf.Variable(xavier_init([h_dim, orig_features]))
    G_b3 = tf.Variable(tf.zeros(shape=[orig_features]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN functions
    # Generator
    def generator(x, m):
        logging.debug("generator called")
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        logging.debug("discriminator called")
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
    # Hat_X = X * M + G_sample * (1-M)
    # because of lags
    Hat_X = X[:, :orig_features] * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1.0 - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    # MSE_loss = \
    # tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

    # because of lags
    MSE_loss = tf.reduce_mean((M * X[:, :orig_features] - M * G_sample) ** 2) / tf.reduce_mean(M)

    # ACC_loss = \
    # tf.reduce_mean((M * X - M * D_prob)) / tf.reduce_mean(M)

    # ACC_loss = \
    # tf.keras.metrics.Accuracy()

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("starting iterations \n")
    logging.debug("Starting iterations")

    # Start Iterations
    # for it in tqdm(range(iterations)):
    for it in range(int(iterations)):

        # Sample batch
        logging.debug("****************************")
        logging.debug(str(it))
        logging.debug("read from file partial reads")
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors
        # H_mb_temp = binary_sampler(hint_rate, batch_size, dim)

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        # because of lags
        M_mb = M_mb[:, :orig_features]

        # because of lags
        H_mb_temp = binary_sampler(hint_rate, batch_size, orig_features)
        H_mb = M_mb * H_mb_temp

        logging.debug("start training of discriminator")
        # _, D_loss_curr = sess.run([D_solver, D_loss_temp],
        #                           feed_dict = {M: M_mb, X: X_mb, H: H_mb})
        _, D_loss_curr, D_prob_curr = sess.run([D_solver, D_loss_temp, D_prob], feed_dict={M: M_mb, X: X_mb, H: H_mb})
        # _, D_loss_curr, ACC_loss_curr = sess.run([D_solver, D_loss_temp, ACC_loss],
        #                           feed_dict = {M: M_mb, X: X_mb, H: H_mb})
        # logging.debug(str(D_loss_curr))
        ACC_loss_curr = classification_accuracy(D_prob_curr, M_mb)
        logging.debug("******************************ACC_loss_curr***************")
        logging.debug(str(ACC_loss_curr))

        # logging.debug(str(np.around(D_prob_curr)))
        # logging.debug(str(np.around(M_mb)))
        # logging.debug(str(np.around(D_prob_curr) == np.around(M_mb)))

        logging.debug("******************************ACC_loss_curr***************")

        logging.debug("start training of generator")
        _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict={X: X_mb, M: M_mb, H: H_mb})
        # _, G_loss_curr, MSE_loss_curr, ACC_loss_curr = \
        # sess.run([G_solver, G_loss_temp, MSE_loss, ACC_loss],
        #          feed_dict = {X: X_mb, M: M_mb, H: H_mb})

    print("finished starting iterations \n")
    logging.debug("*******************************************")
    logging.debug("finished training iterations")
    # save generator as pickle
    saver = tf.train.Saver()
    saver.save(sess, "./model/my-test-model")
    # save the model as txt
    tmp = sess.run(G_W1)
    np.savetxt("weights/G_W1", tmp, delimiter=",")
    tmp = sess.run(G_b1)
    np.savetxt("weights/G_b1", tmp, delimiter=",")

    tmp = sess.run(G_W2)
    np.savetxt("weights/G_W2", tmp, delimiter=",")
    tmp = sess.run(G_b2)
    np.savetxt("weights/G_b2", tmp, delimiter=",")

    tmp = sess.run(G_W3)
    np.savetxt("weights/G_W3", tmp, delimiter=",")
    tmp = sess.run(G_b3)
    np.savetxt("weights/G_b3", tmp, delimiter=",")
