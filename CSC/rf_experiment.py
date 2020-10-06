import smtplib
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
file_handler = logging.FileHandler(filename='experiment_log.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

dataset = pd.read_csv('tep_extended_dataset_simrun1.csv',index_col=False)
dataset_X = dataset.drop(columns='faultNumber').values
dataset_Y = dataset['faultNumber'].values

param_grid = [{'n_estimators': [100, 200, 500],
               'max_features': ['auto', 'log2'],
               'max_depth' : [5,10,50,100,None],
               'criterion' :['gini', 'entropy']}]

RF_clf_gs = GridSearchCV(estimator = RandomForestClassifier(), param_grid=param_grid, scoring='f1_weighted',n_jobs=4, cv=10)
scaler = StandardScaler()
scaler.fit(dataset_X)
scaled_dataset_X = scaler.transform(dataset_X) 

RF_clf_gs.fit(scaled_dataset_X, dataset_Y)
means = RF_clf_gs.cv_results_['mean_test_score']
stds = RF_clf_gs.cv_results_['std_test_score']
logger.info('RF 10CV f1 score mean with 95% confidence interval : ')
for mean, std, params in zip(means, stds, RF_clf_gs.cv_results_['params']):
    logger.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
content = 'Subject: %s\n\n%s' % ('script', 'RF script finished')
mail = smtplib.SMTP('smtp.gmail.com',587)
mail.ehlo()
mail.starttls()
mail.login('notifyme421@gmail.com','sfm1NIUDgv4YtaXV44rk')
mail.sendmail('notifyme421@gmail.com','palusko@gmail.com',content) 
mail.close()
logger.info('FINISHED')
