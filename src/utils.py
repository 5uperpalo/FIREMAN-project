from math import ceil
from typing import List, Literal, Union, Optional

import numpy as np
from lightgbm import Dataset
from numpy import ndarray
from pandas.core.frame import DataFrame
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
from scipy.misc import derivative
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper, gen_features
from torchmetrics import Accuracy, F1Score, Precision, Recall


def scaler_mapper(
    cont_cols: List[str],
    target_col: str,
    identifier: str,
    scaler_mapper_def: Union[dict, None] = None,
):
    """Function that maps scaler functions to appropriate columns. By default assigns scaler to continuous feature columns
    . This behavior can be changed by scaler_mapper_def.
    Only columns defined in mapper object will be present in the transformed dataset.

    Args:
        cont_cols (list): list of continuousl feature columns in the dataset
        target_col (str): target column
        identifier (str): identifier column
        scaler_mapper_def (dict): optional dictionary that contains keys ['cont_cols', 'target_col',
            'identifier_col'] with their corresponding scaler functions from sklearn library

    Returns:
        scaler_mapper (DataFrameMapper): scaler object mapping sklearn scalers to columns in pandas dataframe
    """
    if scaler_mapper_def is None:
        cont_cols_def = gen_features(
            columns=list(map(lambda x: [x], cont_cols)), classes=[StandardScaler]
        )

        target_col_def = [([target_col], None, {})]
        identifier_def = [([identifier], None, {})]

    else:
        cont_cols_def = gen_features(
            columns=list(map(lambda x: [x], cont_cols)),
            classes=[scaler_mapper_def["cont_cols"]],
        )

        target_col_def = [([target_col], scaler_mapper_def["target_col"], {})]
        identifier_def = [([identifier], scaler_mapper_def["identifier_col"], {})]

    scaler_mapper = DataFrameMapper(
        cont_cols_def + target_col_def + identifier_def, df_out=True
    )
    return scaler_mapper


def optimize_df(df: DataFrame, identifier: str, verbose: bool = True):
    """Simple function to assign approporiate columns data types in pandas DataFrame

    Args:
        df (DataFrame): dataset
        identifier (str): identifier column
        cat_cols (list): list of categorical feature columns in the dataset
        verbose (boolean): option to show reduced memory usage

    Returns:
        data (DataFrame): optimized dataset
    """
    data = df.convert_dtypes()
    data[identifier] = data[identifier].astype(str)
    if verbose:
        reduction = (
            1 - (data.memory_usage(deep=True).sum() / df.memory_usage(deep=True).sum())
        ) * 100
        print(f"Memory usage reduced by {reduction:0.2f}%")
    return data


class LGBM_custom_score:
    """Class defining evaluation scores in case fobj, ie. focal loss is defined in LighGBM model training.
    From [documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html):
    'The predicted values. If fobj is specified, predicted values are returned before any transformation,
    e.g. they are raw margin instead of probability of positive class for binary task in this case.'
    """

    def __init__(self, n_class: int):
        self.n_class = n_class

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _prediction(self, preds_raw: ndarray, lgbDataset: Dataset):
        """Helper function to convert raw margin predictions through a
        sigmoid to represent a probability.

        Args:
            preds_raw (ndarray): predictions
            lgbDataset (lightgbm.Dataset): dataset, containing labels, used for prediction

        Returns:
            (y_true, preds): tuple containg labels and predictions for further evaluation
        """
        y_true = lgbDataset.get_label()
        n_example = len(y_true)
        preds = self._sigmoid(preds_raw)

        if self.n_class == 2:
            preds = [int(p > 0.5) for p in preds]
        elif self.n_class > 2:
            preds = preds.reshape(self.n_class, n_example).T
            preds = preds.argmax(axis=1)
        else:
            raise ValueError("n_classes must be int >=2!")

        return y_true, preds

    def _focal_loss(self, y_pred, y_true, alpha, gamma):
        preds = self._sigmoid(y_pred)
        loss = (
            -(alpha * y_true + (1 - alpha) * (1 - y_true))
            * ((1 - (y_true * preds + (1 - y_true) * (1 - preds))) ** gamma)
            * (y_true * np.log(preds) + (1 - y_true) * np.log(1 - preds))
        )
        return loss

    def lgbm_focal_loss(
        self, preds_raw: ndarray, lgbDataset: Dataset, alpha: float, gamma: float
    ):
        """Adapation of the Focal Loss for lightgbm to be used as training loss.
        See original paper:
        * https://arxiv.org/pdf/1708.02002.pdf
        and custom training loss documentation:
        * https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html

        Args:
            y_pred (ndarray): array with the predictions
            dtrain (Dataset): training dataset
            alpha (float): loss function variable
            gamma (float): loss function variable

        Returns:
            grad (float): The value of the first order derivative (gradient) of the loss with
                respect to the elements of preds for each sample point.
            hess (float): The value of the second order derivative (Hessian) of the loss with
                respect to the elements of preds for each sample point.
        """
        y_true = lgbDataset.label
        # N observations x num_class arrays
        if self.n_class > 2:
            y_true = np.eye(self.n_class)[y_true.astype("int")]
            y_pred = preds_raw.reshape(-1, self.n_class, order="F")
        else:
            y_pred = preds_raw.astype("int")

        partial_fl = lambda x: self._focal_loss(x, y_true, alpha, gamma)
        grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
        hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
        if self.n_class > 2:
            return grad.flatten("F"), hess.flatten("F")
        else:
            return grad, hess

    def lgbm_focal_loss_eval(
        self, preds_raw: ndarray, lgbDataset: Dataset, alpha: float, gamma: float
    ):
        """Adapation of the Focal Loss for lightgbm to be used as evaluation loss.
        See original paper https://arxiv.org/pdf/1708.02002.pdf

        Args:
            y_pred (ndarray): array with the predictions
            dtrain (Dataset): training dataset
            alpha (float): loss function variable
            gamma (float): loss function variable

        Returns:
        """
        y_true = lgbDataset.label
        # N observations x num_class arrays
        if self.n_class > 2:
            y_true = np.eye(self.n_class)[y_true.astype("int")]
            y_pred = preds_raw.reshape(-1, self.n_class, order="F")
        else:
            y_pred = preds_raw

        loss = self._focal_loss(y_pred, y_true, alpha, gamma)
        result = ("focal_loss", np.mean(loss), False)
        return result

    def lgbm_f1(self, preds_raw: ndarray, lgbDataset: Dataset):
        """Implementation of the f1 score to be used as evaluation score for lightgbm
        see feval [documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html).
        The adaptation is required since when using custom losses
        the row prediction needs to passed through a sigmoid to represent a
        probability.

        Args:
            preds (ndarray): predictions
            lgbDataset (lightgbm.Dataset): dataset, containing labels, used for prediction

        Returns:
            result (tuple): tuple containing name of the score, its value and bool value for LighGBM (is_higher_better)
        """
        y_true, preds = self._prediction(preds_raw=preds_raw, lgbDataset=lgbDataset)
        result = ("f1", f1_score(y_true, preds, average="weighted"), True)
        return result

    def lgbm_precision(self, preds_raw: ndarray, lgbDataset: Dataset):
        """Implementation of the precision score to be used as evaluation
        score for lightgbm. The adaptation is required since when using custom losses
        the row prediction needs to passed through a sigmoid to represent a
        probability.

        Args:
            preds (ndarray): predictions
            lgbDataset (lightgbm.Dataset): dataset, containing labels, used for prediction

        Returns:
            result (tuple): tuple containing name of the score, its value and bool value for LighGBM (is_higher_better)
        """
        y_true, preds = self._prediction(preds_raw=preds_raw, lgbDataset=lgbDataset)
        result = ("precision", recall_score(y_true, preds, average="weighted"), True)
        return result

    def lgbm_recall(self, preds_raw: ndarray, lgbDataset: Dataset):
        """Implementation of the recall score to be used as evaluation
        score for lightgbm. The adaptation is required since when using custom losses
        the row prediction needs to passed through a sigmoid to represent a
        probability.

        Args:
            preds (ndarray): predictions
            lgbDataset (lightgbm.Dataset): dataset, containing labels, used for prediction

        Returns:
            result (tuple): tuple containing name of the score, its value and bool value for LighGBM (is_higher_better)
        """
        y_true, preds = self._prediction(preds_raw=preds_raw, lgbDataset=lgbDataset)
        result = ("recall", precision_score(y_true, preds, average="weighted"), True)
        return result

    def lgbm_accuracy(self, preds_raw: ndarray, lgbDataset: Dataset):
        """Implementation of the accuracy score to be used as evaluation
        score for lightgbm. The adaptation is required since when using custom losses
        the row prediction needs to passed through a sigmoid to represent a
        probability.

        Args:
            preds (ndarray): predictions
            lgbDataset (lightgbm.Dataset): dataset, containing labels, used for prediction

        Returns:
            result (tuple): tuple containing name of the score, its value and bool value for LighGBM (is_higher_better)
        """
        y_true, preds = self._prediction(preds_raw=preds_raw, lgbDataset=lgbDataset)
        result = ("accuracy", accuracy_score(y_true, preds), True)
        return result


class dl_design:
    """Class with predefined deep learning hidden layer architectures. Especially usefull during
    hyper parameter tuning using Weights&Biases and RayTune to track effect architecture design
    on metrics. Predefined architecture designs are : ["funnel", "pipe", "anti_autoencoder",
    "trapezoid", "anti_trapezoid", "adj_funnel", "apollo"].

    Args:
        input_layer (int): size of input layer
        n_hidden_layers (int): number of hidden layers
        output_layer (int): size of input layer
        design (str): type of design

    Returns:
        hidden_layers (list): list of hidden layers
    """

    def __init__(
        self,
        input_layer: int,
        n_hidden_layers: int,
        output_layer: int,
        design: Literal[
            "funnel",
            "pipe",
            "anti_autoencoder",
            "trapezoid",
            "anti_trapezoid",
            "adj_funnel",
            "apollo",
        ] = "funnel",
    ):
        self.design = design
        self.input_layer = input_layer
        self.n_hidden_layers = n_hidden_layers
        self.output_layer = output_layer

    def __repr__(self):
        return str(self.__dict__)

    def hidden_layers(self):
        if self.design == "funnel":
            return np.linspace(
                self.input_layer * 2,
                self.output_layer,
                self.n_hidden_layers,
                endpoint=False,
                dtype=int,
            ).tolist()

        if self.design == "pipe":
            return [self.input_layer] * self.n_hidden_layers

        if self.design == "anti_autoencoder":
            anti_autoencoder = np.linspace(
                self.input_layer,
                self.input_layer * 2,
                ceil(self.n_hidden_layers / 2),
                dtype=int,
            ).tolist()
            anti_autoencoder.extend(anti_autoencoder[-2::-1])
            return anti_autoencoder

        if self.design == "trapezoid":
            trapezoid = np.array(
                [round(self.input_layer * 1.25)] * self.n_hidden_layers
            )
            trapezoid[[0, -1]] = self.input_layer
            return trapezoid.tolist()

        if self.design == "anti_trapezoid":
            anti_trapezoid = np.array(
                [round(self.input_layer * 0.75)] * self.n_hidden_layers
            )
            anti_trapezoid[[0, -1]] = self.input_layer
            return anti_trapezoid.tolist()

        if self.design == "adj_funnel":
            adj_funnel = np.linspace(
                self.input_layer * 2,
                self.output_layer,
                self.n_hidden_layers,
                endpoint=False,
                dtype=int,
            ).tolist()
            adj_funnel.insert(0, self.input_layer)
            return adj_funnel

        if self.design == "apollo":
            return np.linspace(
                self.input_layer, self.input_layer * 2, self.n_hidden_layers, dtype=int
            ).tolist()


def dl_train_prep(
    data_train: DataFrame,
    data_valid: DataFrame,
    identifier: str,
    target_col: str,
    cont_cols: Optional[list] = None,
    cat_cols: Optional[list] = None,
    embedding_rule: str = "fastai_old",
):
    """Aggregator method to prepare the data for deep models trained in pytorch-widedeep library.
    DISCLAIMER!!!
    This method uses latest - not merged, additions to pytorch_widedeep library.

    Args:
        identifier (str): identifier column
        data_train (DataFrame): training dataset
        data_valid (DataFrame): validation dataset
        cont_cols (list): list of conitunous feature columns in the dataset
        target_col (str): column with predicted value

    Returns:
        X_train (dict): training dataset dictionary
        X_valid (dict): validation dataset dictionary
        tab_preprocessor (TabPreprocessor): deep tabular dataset preprocessor
    """
    tab_preprocessor = TabPreprocessor(
        embedding_rule=embedding_rule,
        embed_cols=cat_cols,
        continuous_cols=cont_cols,
        shared_embed=False,
        scale=False,
    )

    X_tab_train = tab_preprocessor.fit_transform(data_train.drop(columns=[identifier]))
    X_tab_valid = tab_preprocessor.transform(data_valid.drop(columns=[identifier]))

    Y_train = data_train[target_col].values
    Y_valid = data_valid[target_col].values

    X_train = {"X_tab": X_tab_train, "target": Y_train}
    X_valid = {"X_tab": X_tab_valid, "target": Y_valid}

    return X_train, X_valid, tab_preprocessor


def dl_metrics(
    n_classes: Union[int, None] = None,
):
    """Auxiliar method to define metrics tracked during trining of deep learning models.

    Args:
        n_classes (int): number of classes in case of tasks ['binary', 'multiclass']

    Returns:
        metrics_list (list): list of metrics tracked during training of deep learning model
    """
    if n_classes > 2:
        task = "multiclass"
    else:
        task = "binary"
    accuracy = Accuracy(average=None, task=task, num_classes=n_classes)
    precision = Precision(average="micro", task=task, num_classes=n_classes)
    f1 = F1Score(average=None, task=task, num_classes=n_classes)
    recall = Recall(average=None, task=task, num_classes=n_classes)

    metrics_list = [accuracy, precision, f1, recall]
    return metrics_list


def dl_predict(
    data: DataFrame,
    model: WideDeep,
    tab_preprocessor: TabPreprocessor,
    wide_preprocessor: Union[WidePreprocessor, None] = None,
):
    """Aggregator method to predict target value from pandas Dataframe using pretrained deep learning model.

    Args:
        model (WideDeep): pretained model
        tab_preprocessor (TabPreprocessor): deep tabular dataset preprocessor
        wide_preprocessor (WidePreprocessor): wide tabular dataset preprocessor

    Returns:
        preds (ndarray): predictions
    """
    if wide_preprocessor:
        X_wide = wide_preprocessor.transform(data)
    else:
        X_wide = None
    X_tab = tab_preprocessor.transform(data)
    preds = model.predict(X_wide=X_wide, X_tab=X_tab)
    return preds
