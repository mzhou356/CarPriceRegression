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

