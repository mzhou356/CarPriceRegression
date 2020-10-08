"""
This module contains helper functions for data_process.py.
"""
import pickle as pkl

def load_dict(filename):
    with open(filename, "rb") as handle:
        dict_mapping = pkl.load(handle)
    return dict_mapping

def binning(col, thresholds):
    """
    This function bins numerical features into categories.

    Args:
    col: a numeric value for the feature column
    thresholds: a dictionary for map between threshold and categorical value
    df: a pandas dataframe, the original data.

    return:
    transformed col as a categorical object type.
    """
    for k, v in thresholds.items():
        if k[0] <= col <= k[1]:
            return v
    return None

def input_process(user_input,maps):
    cols = ["year","mileage","mpg","engineSize"]
    binned_columns = ["binned_year","mil_cat","binned_mpg","engine_binned"]
    processed = {}
    for i, col in enumerate(cols):
        original_value = user_input[col]
        converted_value = binning(original_value, maps[i])
        processed[binned_columns[i]] = converted_value
    return processed
    
        