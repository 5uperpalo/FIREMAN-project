"""Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
"""

# Necessary packages
import numpy as np
import tensorflow as tf


def normalization(data):
    """Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    """

    print("starting normalization in function \n")
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
        # norm_data[:,i] = norm_data[:,i] / (max_val[i] - min_val[i] + 1e-6)
        # norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
        # norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)

    # Return norm_parameters for renormalization
    norm_parameters = {"min_val": min_val, "max_val": max_val}

    print("finishing normalization in function \n")
    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    """Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    """

    print("starting renormalization function \n")
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = (renorm_data[:, i] * (max_val[i] - min_val[i] + 1e-6)) + min_val[i]
        # renorm_data[:,i] = renorm_data[:,i] + min_val[i]
        # renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)
        # renorm_data[:,i] = renorm_data[:,i] + min_val[i]

    print("finishing renormalization function \n")
    return renorm_data


def rounding(imputed_data, data_x):
    """Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    """

    print("starting rounding function \n")
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    print("finishing rounding function \n")
    return rounded_data


def rmse_loss(ori_data, imputed_data, data_m):
    """Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values
      - imputed_data: imputed data
      - data_m: indicator matrix for missingness

    Returns:
      - rmse: Root Mean Squared Error
    """

    print("starting rmse_loss function \n")
    ori_data, _ = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    print("finishing rmse_loss function \n")
    return rmse


def xavier_init(size):
    """Xavier initialization.

    Args:
      - size: vector size

    Returns:
      - initialized random vector.
    """
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


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


def sample_always_specific_sensor(rows, cols, sens_no):
    sampled_matrix = np.ones((rows, cols))
    sampled_matrix[:, sens_no] = 0
    return sampled_matrix


def binary_sampler_with_lags(p, rows, cols, orig_features):
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, orig_features])
    unif_random_matrix_add_zeros = np.zeros((rows, cols - orig_features))
    unif_random_matrix = np.concatenate((unif_random_matrix, unif_random_matrix_add_zeros), axis=1)
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def binary_sampler_with_lags_old(p, rows, cols, orig_features):
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    unif_random_matrix[:, orig_features:] = 0.0
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    """Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    """
    return np.random.uniform(low, high, size=[rows, cols])


def sample_batch_index(total, batch_size):
    """Sample index of the mini-batch.

    Args:
      - total: total number of samples
      - batch_size: batch size

    Returns:
      - batch_idx: batch index
    """
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx
