#!/usr/bin/env python3

### This script contains helper function to process data for carPrice regression model 

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


def readData(dirpath,ext):
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
            df = pd.read_csv(dirpath+filename)
            df["brand"] = brand
            df.columns = ['model', 'year', 'price', 'transmission', 
                          'mileage', 'fuelType', 'tax','mpg', 'engineSize',"brand"]
            datas.append(df)
    return pd.concat(datas,axis=0)

def EDA_CAT_func(colname,target,df,fontsize, figsize,layout,boxplot=False):
    """
    This function performs basic exploratory data analyses on a specific feature column,
    This column needs to be categorical in nature. 
    
    Args:
    colname: a string, the feature column name. 
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
    groupedDF = df.groupby(colname)
    count_df = groupedDF[target].count().reset_index().sort_values(target)
    if boxplot:
        groupedDF.boxplot(column=target,fontsize=fontsize,
                                       figsize=figsize,layout=layout)
        plt.show()
    groupedDF[target].median().reset_index().sort_values("price").plot.scatter(x=colname,y=target,rot=90,figsize=figsize)
    plt.show()
    return count_df

def binning_func(col,thresholds):
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
        if col>=k[0] and col<=k[1]:
            return v
    
    