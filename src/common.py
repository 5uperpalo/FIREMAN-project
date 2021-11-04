import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
import json

def intsec(list1, list2):
    """Simple intesection of two lists.

    Args:
        list1 (list): list1
        list2 (list): list2

    Returns:
        list (list): intersection of lists
    """
    return list(set.intersection(set(list1), set(list2)))


def diff(list1, list2):
    """Simple difference of items in list2 from list1.

    Args:
        list1 (list): list1
        list2 (list): list2

    Returns:
        list (list): difference of lists
    """
    return list(set(list1).difference(set(list2)))


def scale(data_pd, non_scale_cols, scaler_sk='Standard'):
    """Procedure to scale the dataset except the given list of columns.

    Args:
        data_pd (obj): pandas dataframe
        non_scale_cols (list): columns to not scale
        scaler_sk (str, sklearn.peprocessing obj): type of scaler from['Standard', 'Yeo-Johnson',
        'Robust', 'MinMax'] or already fitted scaler

    Returns:
        tuple (tuple): data_pd_scaled (obj): scaled pandas dataframe\n
        scaler_sk (obj): sklearn scaler object
    """
    non_scale_cols = intsec(data_pd.columns.values, non_scale_cols)
    data_pd_toscale = data_pd.drop(columns=non_scale_cols)
    if type(scaler_sk) is str:
        if scaler_sk == 'Standard':
            scaler_sk = StandardScaler()
        elif scaler_sk == 'Yeo-Johnson':
            scaler_sk = PowerTransformer(method='yeo-johnson')
        elif scaler_sk == 'Robust':
            scaler_sk = RobustScaler()
        elif scaler_sk == 'MinMax':
            scaler_sk = MinMaxScaler()
        scaler_sk.fit(data_pd_toscale)
    # if 'sklearn.peprocessing' in str(type(scaler_sk)):

    data_pd_scaled = pd.DataFrame(scaler_sk.transform(data_pd_toscale),
                                  columns=data_pd_toscale.columns.values)
    data_pd_scaled[non_scale_cols] = data_pd[non_scale_cols].copy()
    return data_pd_scaled, scaler_sk

def json_load(file_loc):
    """Helper function to open/close json file, otherwise the python outputs warning that the file remains opened

    Args:
        file_loc (str): location of the file

    Returns:
        file_content (dict): content of json file in dict
    """
    with open(file_loc, "rb") as f:
        file_content = json.load(f)
    return file_content