# Adjusted version of "Generative Adversarial Imputation Networks (GAIN)"

Adjusted from https://github.com/jsyoon0823/GAIN/blob/master/utils.py <\br>
You may find example usage with tf -> tf2 adjustment procedure in fireman_gain.ipynb

usage:
```
from GAIN.gain_tf2 import gain
from GAIN.utils_tf2 import binary_sampler
from GAIN.utils_tf2 import rmse_loss

tep_dataset = pd.read_csv('Tennessee_Event-Driven/datasets/dataset.csv',index_col=False)

dataset_X = tep_dataset.drop(columns=['faultNumber', 'simulationRun', 'sample']).values
dataset_Y = tep_dataset['faultNumber'].values

no, dim = dataset_X.shape

# drop value probability
p = 0.1

# Introduce missing data
mask = binary_sampler(1-p, no, dim)
dataset_X_missing = dataset_X.copy()
dataset_X_missing[mask == 0] = np.nan

gain_parameters = {'batch_size': 128,
                 'hint_rate': 1.5,
                 'alpha': 100,
                 'iterations': 1000}
                 
dataset_X_imputed = gain(dataset_X_missing, gain_parameters)

rmse_loss(dataset_X, dataset_X_imputed, mask)
```
