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
            df = pd.read_csv(dirpath+filename)
            df.columns = ['model', 'year', 'price', 'transmission', 
                          'mileage', 'fuelType', 'tax','mpg', 'engineSize']
            datas.append(df)
    return pd.concat(datas,axis=0)

def EDA_CAT_func(colname,target,df,fontsize, figsize,layout):
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
    groupedDF.boxplot(column=target,fontsize=fontsize,
                                       figsize=figsize,layout=layout)
    plt.show()
    groupedDF[target].median().reset_index().plot.scatter(x=colname,y=target)
    plt.show()
    return count_df