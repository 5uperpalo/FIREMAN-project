from copy import deepcopy
import pandas as pd
import numpy as np
import multiprocessing
import dill

from . import common
from torch.optim import SGD, lr_scheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pytorch_widedeep.dataloaders import DataLoaderImbalanced, DataLoaderDefault
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.initializers import XavierNormal
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint

from torchmetrics import F1 as F1_torchmetrics
from torchmetrics import Accuracy as Accuracy_torchmetrics
from torchmetrics import Precision as Precision_torchmetrics
from torchmetrics import Recall as Recall_torchmetrics

# use_gpu = True
# use_cuda = use_gpu and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

# torch.set_num_threads(multiprocessing.cpu_count())
# torch.set_num_interop_threads(multiprocessing.cpu_count())


class dl_design:
    """Object used to define different DL network designs"""
    def __init__(
        self,
        input_layer: int,
        n_hidden_layers: int,
        output_layer: int,
        design: str = "funnel",
    ):
        self.design = design
        self.input_layer = input_layer
        self.n_hidden_layers = n_hidden_layers
        self.output_layer = output_layer

    def __repr__(self):
        return str(self.__dict__)

    def hidden_layers(self):
        if self.design == "funnel":
            return np.linspace(self.input_layer * 2,
                               self.output_layer,
                               self.n_hidden_layers,
                               endpoint=False, dtype=int).tolist()

        if self.design == "pipe":
            return [self.input_layer]*self.n_hidden_layers

        if self.design == "anti_autoencoder":
            anti_autoencoder = np.linspace(self.input_layer, self.input_layer*2, ceil(self.n_hidden_layers/2), dtype=int).tolist()
            anti_autoencoder.extend(anti_autoencoder[-2::-1])
            return anti_autoencoder

        if self.design == "trapezoid":
            trapezoid = np.array([round(self.input_layer*1.25)]*self.n_hidden_layers)
            trapezoid[[0, -1]] = self.input_layer
            return trapezoid.tolist()

        if self.design == "anti_trapezoid":
            anti_trapezoid = np.array([round(self.input_layer*0.75)]*self.n_hidden_layers)
            anti_trapezoid[[0, -1]] = self.input_layer
            return anti_trapezoid.tolist()

        if self.design == "adj_funnel":
            adj_funnel = np.linspace(self.input_layer*2, self.output_layer, self.n_hidden_layers, endpoint=False, dtype=int).tolist()
            adj_funnel.insert(0, input_layer)
            return adj_funnel

        if self.design == "apollo":
            return np.linspace(self.input_layer, self.input_layer*2, self.n_hidden_layers, dtype=int).tolist()


def pipeline_data_train_prep(
    data,
    test_size_train,
    test_size_valid,
    cat_cols,
    scaler_def,
    random_state,
    identifier,
    target,
    verbose=True,
):
    """Procedure to prepare the data for processing by models in the training pipeline.

    Args:
        data (pandas): training dataset
        test_size_train (float): fraction of the training dataset used for validation and testing
        test_size_valid (float): fraction of the validation and testing dataset used for testing
        cat_cols (list): list of catgorical columna names in the dataset
        scaler_def (str): type of scaler to use, possible choices from sklearn library
        random_state (int): random state to make the results repeatable
        identifier (string): column with identifier IDs
        target (string): column with labels
        verbose (boolean): wheter to show print in the output

    Returns:
        data_train_scaled (pandas): scaled training dataset
        data_valid_scaled (pandas): scaled validation dataset
        data_test_scaled (pandas): scaled validation dataset, without transformed target variable
        cont_cols (list): list of continuous column names in the dataset
        scaler (obj): scaler
    """
    # some categorical column names might contain ".", this create an issue when the feature is
    # one-hot-encodded in pytorch-widedeep, error log:
    # KeyError: 'module name can\'t contain ".", got: emb_layer_battlepass_8008.0'
    for i, col in enumerate(cat_cols):
        if "." in col:
            cat_cols[i] = col.replace(".", "_")
            data.rename(columns={col: col.replace(".", "_")}, inplace=True)

    if verbose:
        print(
            "Size of dataset classes:\n{}".format(
                data[target].value_counts()
            )
        )

    cont_cols = common.diff(data.drop(columns=[identifier,target]).columns.values, cat_cols)

    data_train, data_valid = train_test_split(
        data,
        test_size=test_size_train,
        stratify=data[target],
        random_state=random_state,
    )
    data_valid, data_test = train_test_split(
        data_valid,
        test_size=test_size_valid,
        stratify=data_valid[target],
        random_state=random_state,
    )

    data_train.reset_index(inplace=True, drop=True)
    data_valid.reset_index(inplace=True, drop=True)
    data_test.reset_index(inplace=True, drop=True)

    # data scale
    data_train_scaled, Scaler = common.scale(
        data_train, cat_cols + [target, identifier], scaler_sk=scaler_def
    )
    data_valid_scaled, Scaler = common.scale(
        data_valid, cat_cols + [target, identifier], scaler_sk=Scaler
    )
    data_test_scaled, Scaler = common.scale(
        data_test, cat_cols + [target, identifier], scaler_sk=Scaler
    )

    return data_train_scaled, data_valid_scaled, data_test_scaled, cont_cols, Scaler


def dl_model_data_prep(data_train, data_valid, cat_cols, cont_cols, target):
    """Procedure to prepare data for DL model training.

    Args:
        data_train (pandas): scaled training dataset
        data_valid (pandas): scaled validation dataset
        cat_cols (list): list of catgorical column names in the dataset
        cont_cols (list): list of continuous column names in the dataset
        target (string): column with labels

    Returns:        
        X_train (dict): training dataset
        X_val (dict): validation dataset
        wide_preprocessor (obj): DL model preprocessor for categorical columns
        tab_preprocessor (obj): DL model preprocessor for continuous columns
    """
    if cat_cols:
        wide_preprocessor = WidePreprocessor(wide_cols=cat_cols)
        X_wide_train = wide_preprocessor.fit_transform(data_train)
        X_wide_valid = wide_preprocessor.transform(data_valid)

        tab_preprocessor = TabPreprocessor(
            embedding_rule="fastai_old",
            embed_cols=cat_cols,
            continuous_cols=cont_cols,
            shared_embed=False,
            scale=False,
        )
    else:
        wide_preprocessor = None
        tab_preprocessor = TabPreprocessor(
            continuous_cols=cont_cols,
            shared_embed=False,
            scale=False,
        )

    X_tab_train = tab_preprocessor.fit_transform(data_train)
    X_tab_valid = tab_preprocessor.transform(data_valid)

    Y_train = data_train[target].values
    Y_valid = data_valid[target].values
    
    if cat_cols:
        X_train = {"X_wide": X_wide_train, "X_tab": X_tab_train, "target": Y_train}
        X_val = {"X_wide": X_wide_valid, "X_tab": X_tab_valid, "target": Y_valid}
    else:
        X_train = {"X_tab": X_tab_train, "target": Y_train}
        X_val = {"X_tab": X_tab_valid, "target": Y_valid}
    
    return X_train, X_val, wide_preprocessor, tab_preprocessor


def dl_train(X_train, X_val, wide_preprocessor, tab_preprocessor, task, verbose):
    """Procedure to train and validate the DL model for classification.

    Args:
        X_train (dict): training dataset
        X_val (dict): validation dataset
        wide_preprocessor (obj): DL model preprocessor for categorical columns
        tab_preprocessor (obj): DL model preprocessor for continuous columns
        task (str): if it is binary or multiclass classification task
        verbose (boolean): option to show progress of classification model optimization

    Returns:
        model (obj): DL model
    """

    n_classes = np.unique(X_train["target"]).size

    accuracy = Accuracy_torchmetrics(average=None, num_classes=n_classes)
    precision = Precision_torchmetrics(average="micro", num_classes=n_classes)
    f1 = F1_torchmetrics(average=None, num_classes=n_classes)
    recall = Recall_torchmetrics(average=None, num_classes=n_classes)
    metrics = [accuracy, precision, f1, recall]

    input_layer = len(tab_preprocessor.continuous_cols)

    if wide_preprocessor:
        for i in tab_preprocessor.embed_dim.values():
            input_layer += i

        if task == "binary":
            output_layer = 1
        else:
            output_layer = n_classes

        hidden_layers = dl_design(input_layer, 3, output_layer, design="funnel").hidden_layers()

        wide = Wide(wide_dim=wide_preprocessor.wide_dim, pred_dim=output_layer)

        deeptabular = TabMlp(
            mlp_hidden_dims=hidden_layers,
            column_idx=tab_preprocessor.column_idx,
            embed_input=tab_preprocessor.embeddings_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            mlp_batchnorm=True,
            mlp_batchnorm_last=True,
            mlp_linear_first=True,
        )

        model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=output_layer)

        wide_opt = SGD(model.wide.parameters(), lr=0.1)
        deep_opt = SGD(model.deeptabular.parameters(), lr=0.1)
        wide_sch = lr_scheduler.StepLR(wide_opt, step_size=5)
        deep_sch = lr_scheduler.StepLR(deep_opt, step_size=5)

        early_stopping = EarlyStopping()
        model_checkpoint = ModelCheckpoint(save_best_only=True, verbose=int(verbose))

        if task == "binary":
            objective = "binary_focal_loss"
        if task == "multiclass":
            objective = "multiclass_focal_loss"

        trainer = Trainer(
            model,
            objective=objective,
            callbacks=[early_stopping, model_checkpoint],
            lr_schedulers={"wide": wide_sch, "deeptabular": deep_sch},
            initializers={"wide": XavierNormal, "deeptabular": XavierNormal},
            optimizers={"wide": wide_opt, "deeptabular": deep_opt},
            metrics=metrics,
        )
    else:
        if task == "binary":
            output_layer = 1
        else:
            output_layer = n_classes

        hidden_layers = dl_design(input_layer, 3, output_layer, design="funnel").hidden_layers()

        deeptabular = TabMlp(
            mlp_hidden_dims=hidden_layers,
            column_idx=tab_preprocessor.column_idx,
            continuous_cols=tab_preprocessor.continuous_cols,
            mlp_batchnorm=True,
            mlp_batchnorm_last=True,
            mlp_linear_first=True,
        )

        model = WideDeep(deeptabular=deeptabular, pred_dim=output_layer)

        deep_opt = SGD(model.deeptabular.parameters(), lr=0.1)
        deep_sch = lr_scheduler.StepLR(deep_opt, step_size=5)

        early_stopping = EarlyStopping()
        model_checkpoint = ModelCheckpoint(save_best_only=True, verbose=int(verbose))

        if task == "binary":
            objective = "binary_focal_loss"
        if task == "multiclass":
            objective = "multiclass_focal_loss"

        trainer = Trainer(
            model,
            objective=objective,
            callbacks=[early_stopping, model_checkpoint],
            lr_schedulers={"deeptabular": deep_sch},
            initializers={"deeptabular": XavierNormal},
            optimizers={"deeptabular": deep_opt},
            metrics=metrics,
        )
    trainer.fit(
        X_train=X_train,
        X_val=X_val,
        n_epochs=5,
        batch_size=100,
        custom_dataloader=DataLoaderImbalanced,
        oversample_mul=5,
    )
    return trainer


def dl_predict(data, wide_preprocessor, tab_preprocessor, model):
    """Procedure to predict values from pandas using provided dl model

    Args:
        data (pandas): pandas dataframe containing data to predict
        wide_preprocessor (obj): DL model preprocessor for categorical columns
        tab_preprocessor (obj): DL model preprocessor for continuous columns
        model (obj): DL model

    Returns:
       predicted (list): predicted values 
    """
    X_tab = tab_preprocessor.transform(data)
    if wide_preprocessor:
        X_wide = wide_preprocessor.transform(data)
        return model.predict(X_wide=X_wide, X_tab=X_tab)
    else:
        return model.predict(X_tab=X_tab)


def evaluate(actual, predicted):
    """Procedure to print classification report

    Args:
       actual (list): actual values
       predicted (list): predicted values
    """
    print(
        "Classification report:\n{}".format(
            classification_report(actual, predicted)
        )
    )


def train(
    task,
    data,
    column_types_loc,
    parameters,
    save_loc=None,
    verbose=True,
    datasets=True,
):
    """Procedure to sequentially proceed through all pipeline steps to train the DL and ML models.

    Args:
        task (str): if it is binary or multiclass classification task
        data (pandas): training dataframe
        column_types_loc (string): location of json file with columns definitions
        parameters (dict): dictionary with parameters for DL/ML models
        save_loc (str): directory where to save trained models
        verbose (boolean): option to show progress of the model training
        datasets (boolean): whether to return train, valid, test datasets for outlier detection

    Returns:
        models (dict): dictionary with trained models objects
    """
    # identifiers & params
    column_types = common.json_load(column_types_loc)
    identifier = column_types["identifier"]
    cat_cols = column_types["categorical"]
    target = column_types["target"]

    # scalers & models parameters
    test_size_train = parameters["test_size_train"]
    test_size_valid = parameters["test_size_valid"]
    scaler_def = parameters["scaler"]
    random_state = parameters["random_state"]

    (
        data_train_scaled,
        data_valid_scaled,
        data_test_scaled,
        cont_cols,
        Scaler,
    ) = pipeline_data_train_prep(data,
                                 test_size_train,
                                 test_size_valid,
                                 cat_cols,
                                 scaler_def,
                                 random_state,
                                 identifier,
                                 target,
                                 verbose=verbose)

    (
        X_train,
        X_val,
        wide_preprocessor,
        tab_preprocessor
    ) = dl_model_data_prep(data_train_scaled.drop(columns=[identifier]),
                           data_valid_scaled.drop(columns=[identifier]),
                           cat_cols,
                           cont_cols,
                           target)

    trainer = dl_train(X_train,
                       X_val,
                       wide_preprocessor,
                       tab_preprocessor,
                       task,
                       verbose)

    predicted = dl_predict(data_test_scaled.drop(columns=[identifier]),
                           wide_preprocessor,
                           tab_preprocessor,
                           trainer)

    evaluate(data_test_scaled[target].values, predicted)
    
    models = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "scaler": Scaler,
        "dl": trainer,
        "dl_wide_preprocessor": wide_preprocessor,
        "dl_tab_preprocessor": tab_preprocessor,
    }

    if save_loc:
        with open(save_loc, "wb") as f:
            dill.dump(models, f)

    if datasets:
        return data_train_scaled, data_valid_scaled, data_test_scaled, models
    else:
        return models


def predict(data, column_types_loc, models_loc):
    """Procedure for using trained models.

    Args:
        data (pandas): testing/production dataframe
        column_types_loc (string): location of json file with columns definitions
        models_loc (string): location of file that includes dictionary with models objects

    Returns:
        result (pandas): dataframe with Predicted values.
    """
    # identifiers
    column_types = common.json_load(column_types_loc)
    identifier = column_types["identifier"]

    with open(models_loc, "rb") as f:
        models = dill.load(f)

    cat_cols = models["cat_cols"]
    cont_cols = models["cont_cols"]
    Scaler = models["scaler"]
    model = models["dl"]
    wide_preprocessor = models["dl_wide_preprocessor"]
    tab_preprocessor = models["dl_tab_preprocessor"]

    data = data[cat_cols + cont_cols + [identifier]]
    data_scaled, Scaler = common.scale(
        data, cat_cols + [identifier], scaler_sk=Scaler
    )
    predicted = dl_predict(data_scaled.drop(columns=[identifier]),
                           wide_preprocessor,
                           tab_preprocessor,
                           model)

    return predicted
