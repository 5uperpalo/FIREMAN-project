import smtplib
import logging
#from tqdm import tqdm
import pandas as pd
import numpy as np
from GAIN.gain_tf2 import gain
from GAIN.utils_tf2 import binary_sampler
from GAIN.utils_tf2 import rmse_loss
from sklearn.model_selection import ParameterGrid

logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
file_handler = logging.FileHandler(filename='experiment_log.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

dataset = pd.read_csv('tep_extended_dataset_simrun1.csv',index_col=False)
dataset_X = dataset.drop(columns='faultNumber').values
dataset_Y = dataset['faultNumber'].values
no, dim = dataset_X.shape

p_list = [0.25]
param_grid = {'batch_size': [16,32,64,128,256], 
              'hint_rate' : [1,2,5,10,100],
              'alpha' : [1,10,100,1000,10000],
              'iterations' : [10,100,1000,10000],}
grid = ParameterGrid(param_grid)

try:
    dataset_X_rmse_list = []    
    for p in p_list: 
        mask = binary_sampler(1-p, no, dim)
        dataset_X_missing = dataset_X.copy()
        dataset_X_missing[mask == 0] = np.nan
    
        np.savetxt("dataset_X_missing_"+str(p)+".csv", dataset_X_missing, delimiter=",")

        dataset_X_rmse_list_temp = []
        index_names = []

        for params in grid:
            dataset_X_imputed = gain(dataset_X_missing, params)
            loss_temp = rmse_loss(dataset_X, dataset_X_imputed, mask)

            logger.info(params)
            logger.info("RMSE loss:")
            logger.info(loss_temp)

            index_names.append(str(params))
            dataset_X_rmse_list_temp.append(loss_temp)
        dataset_X_rmse_list.append(dataset_X_rmse_list_temp)
        
    result_pd = pd.DataFrame(index = index_names, columns=['p_' + str(p) for p in p_list], data = np.array(dataset_X_rmse_list).T)
    result_pd.to_csv('TEP_GAINImputation_RMSE.csv')

    content = 'Subject: %s\n\n%s' % ('script', 'script finished')
    mail = smtplib.SMTP('smtp.gmail.com',587)
    mail.ehlo()
    mail.starttls()
    mail.login('notifyme421@gmail.com','sfm1NIUDgv4YtaXV44rk')
    mail.sendmail('notifyme421@gmail.com','palusko@gmail.com',content) 
    mail.close()
    logger.info('FINISHED')
except:
    content = 'Subject: %s\n\n%s' % ('script', 'script crashed')
    mail = smtplib.SMTP('smtp.gmail.com',587)
    mail.ehlo()
    mail.starttls()
    mail.login('notifyme421@gmail.com','sfm1NIUDgv4YtaXV44rk')
    mail.sendmail('notifyme421@gmail.com','palusko@gmail.com',content)
    mail.close()
    logger.info('CRASHED')
