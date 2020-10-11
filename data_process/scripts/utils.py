# pylint: disable=too-many-arguments
#!/usr/bin/env python3
"""
This script contains helper functions to process data for car price regression models.
"""
import os
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def read_data(dirpath, ext):
    """
    This function loads in all files with file type ext and merge into one pandas dataframe.

    Args:
    dirpath: a string, directory to the data file path.
    ext: file extension (ex ".csv")

    Returns:
    A pandas dataframe with all merge files.
    """
    datas = []
    for filename in os.listdir(dirpath):
        if filename.endswith(ext):
            brand = filename.split(".")[0].split("_")[-1]
            df = pd.read_csv(dirpath + filename)
            df["brand"] = brand
            df.columns = ['model', 'year', 'price', 'transmission',
                          'mileage', 'fuelType', 'tax', 'mpg', 'engineSize', "brand"]
            datas.append(df)
    return pd.concat(datas, axis=0)

def eda_cat_func(colname, target, df, fontsize, figsize, layout=None, boxplot=False):
    """
    This function performs basic exploratory data analyses on a specific feature column,
    This column needs to be categorical in nature.

    Args:
    colname: a string, the feature column name or a list of strings if group by more than one.
    target: a string, the target column name.
    df: a pandas dataframe, the original data.
    fontsize: fontsize for boxplot
    figsize: figsize for boxplot
    layout: arrangement of row and col for boxplot

    Outputs the count for each category, boxplot for each category, and line charts for
    median trends.

    return:
    returns count by category dataframe.
    """
    grouped_df = df.groupby(colname)
    count_df = grouped_df[target].count().reset_index().sort_values(target)
    count_df.columns = [colname, "counts"]
    if boxplot:
        grouped_df.boxplot(column=target, fontsize=fontsize,
                           figsize=figsize, layout=layout)
        plt.show()
    grouped_df[target].median().reset_index().sort_values(
        colname).plot.scatter(x=colname, y=target, rot=90, figsize=figsize)
    plt.show()
    return count_df

def binning_func(col, thresholds):
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

def chi_square_test(col1, col2):
    """
    This function allows this callable function to be used for pandas corr method parameter.

    Args:
    col1: a string, column name.
    col2: a string, column name.

    Returns:
    pvalue for one way chisquare test.
    """
    return chisquare(col1, col2).pvalue

def cat_feature_corr(df, cols_to_drop, chi_test=False,):
    """
    This function creates a dataframe of correation between categorical features
    using spearman or chisquaretest

    Args:
    df: pandas dataframe.
    chi_test: chisquare or spearman
    cols_to_drop: a list of string (cols to drop that are not categorical features).

    Returns:
    a pandas dataframe of categorical feature correlation.
    """
    df_factored = df.drop(cols_to_drop, axis=1).apply(lambda x: pd.factorize(x)[0]) + 1
    if chi_test:
        result = df_factored.corr(method=chi_square_test)
    else:
        result = df_factored.corr(method="spearman")
    return result

def save_object(filename, python_object):
    """
    This function saves object into a pickle format.

    Args:
    filename: a string, name to save the python object as a pickle object.
    python_object: a dictionary, a list, or any other python objects.
    """
    with open(filename, "wb") as f_handle:
        pkl.dump(python_object, f_handle, protocol=pkl.HIGHEST_PROTOCOL)
