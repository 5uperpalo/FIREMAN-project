# <img src="https://github.com/5uperpalo/FIREMAN-project/blob/master/docs/sources/assets/images/logo-fireman.png" height="64" />FIREMAN-project main repository

Machine learning scripts and notebooks related to [FIREMAN project](https://fireman-project.eu/).

Documentation: https://fireman-project.readthedocs.io

Artificial dataset used for development:
* [TEP explanation](https://medium.com/@mrunal68/tennessee-eastman-process-simulation-data-for-anomaly-detection-evaluation-d719dc133a7f)
* [Extended TEP dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1)

Repositories:
* [Data imputation](https://github.com/5uperpalo/FIREMAN-project_imputation) - data imputation scripts, notebooks and utilities

Scripts and notebooks:
* [00 Preprocessing](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/00_dataset_preprocessing_general_approach.ipynb) - general approach to preprocessing of the dataset with exploratory data science(feature values/distributions/correlations/etc.)
* [01 Classification](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/01_classification.ipynb) - basic application and result analysis of [scikit-learn](https://scikit-learn.org/stable/) classifiers
* [02 Extended TEP](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/02_extended_tep.ipynb) - transformation of Extended TEP dataset in .rdata format to csv
* [03 Clustering](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/03_density-based_analysis.ipynb) - unsupervised density-based clustering analysis of the dataset using DBSCAN and OPTICS
* [04 Stream-based ML - MOA](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/04_moa_analysis.ipynb) - stream-based machine learning using [MOA](https://moa.cms.waikato.ac.nz/)
* [05 Pipeline](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/05_pipeline.ipynb) - machine learning pipeline template
* [06 Stream-based ML - scikit-multiflow](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/06_streamML_and_concept_drift_detection.ipynb) - stream-bades machine learning and concept drift detection using [scikit-multiflow](https://scikit-multiflow.github.io/), project closely related to [MOA](https://moa.cms.waikato.ac.nz/)
* [07 PCA](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/07_PekkaR.ipynb) - dataset analysis using PCA related to a collaborating student masters thesis
  * [08 related analysis](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/08_Tennessee_Variables_PekkaR.ipynb) - dataset analysis related to a collaborating student masters thesis
* [09 PowerConverter dataset](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/09_PowerConverter_dataset_preprocessing.ipynb) - Power Converter dataset preprocessing
* [10 PowerConverter dataset classification using DL](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/10_DL_w_RayTune.ipynb) - Power Converter dataset preprocessing
* [11 PowerConverter dataset classification using LightGBM](https://github.com/5uperpalo/FIREMAN-project/tree/master/notebooks/11_LightGBM_w_RayTune.ipynb) - Power Converter dataset preprocessing

Additional materials:
* [CSC/](https://github.com/5uperpalo/FIREMAN-project/tree/master/CSC) - scripts usable in [CSC](https://research.csc.fi/)
* [workshop_05132020/](https://github.com/5uperpalo/FIREMAN-project/tree/master/workshop_05132020) - materials and presentations related to Fireman ML workshop organized in May 2020
