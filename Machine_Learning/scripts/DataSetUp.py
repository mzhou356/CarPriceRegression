# DataSetUp Class for split dataset 
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf 
from sklearn.model_selection import train_test_split

class DataSetUp:
    def __init__(self,features,label):
        """
        label: a pandas Series, label column
        feature: feature columns without the label column
        val_map: only for categorical embedding, mapping for categorical mapping 
        embed_cols: only for categorical embedding, columns for embedding
        non_embed_cols: only for categorical embedding, columns for non embedded columns 
        """
        self._label = label
        self._features = features
        self._val_map = None
        self._embed_cols = None
        self._non_emebd_cols = None
    
    def data_split(self,seed,test_size,dev_set = False,dev_seed=None,dev_size=None):
        """
        seed: random state for data split 
        test_size: a float, percentage of data for testing 
        dev_set: include dev set or only test and train. 
        dev_seed: random seed for split train data into train and dev 
        dev_size: a float, percentage of data for dev and train 
        
        returns:
        pandas data frame of train, test or train, test, and dev 
        """
        X_train, X_test, y_train, y_test = train_test_split(self._features,self._label, 
                                                            test_size = test_size, random_state=seed)
        return X_train,X_test,y_train,y_test
        if dev_set:
            X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,
                                                       test_size = dev_size,random_state=dev_seed)
            return X_train,X_dev,X_test,y_train,y_dev,y_test 
    
    def make_tensor_dataset(self,X,y,batch_size):
        """
        This function generates tensorflow train and test dataset for NN.
    
        args:
        X: a pandas dataframes, features 
        y: a pandas series, label 
        batch_size: batch size for training 
      
    
        returns:
        tensforflow dataset
        """
        data_set = tf.data.Dataset.from_tensor_slices((X.values,
                 y.values)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(X.shape[0])
        return data_set
    
    def categorical_mapping(self,X_train,embed_cols):
        """
        This function generates categorical map for entity embedding using training dataset 
        
        Args: 
        X_train: a pandas dataframe, training dataset for map categories 
        embed_cols:a list of feature name for embed columns 
        
        """
        self._embed_cols = embed_cols 
        self._non_emebd_cols = [c for c in X_train.columns if c not in embed_cols]
        for c in embed_cols:
            raw_values = X_train[c].unique()
            val_map = {}
            for i in range(len(raw_values)):
                # start with zero so fillna with zero shows the category in new dataset is not in any existing categories) 
                val_map[raw_values[i]]=i+1 
        self._val_map = val_map    
    
    def cate_data_list(self,X):
        """
        This function transforms features using X_train data to match the format to train neural network 
    
        Args:
        X_train: pandas df, features for training 
        X: pandas df, test or dev data set 
       
        Returns:
        a python list of features appropriate for categorical embedding 
        """
        input_list_X = []
        
        for c in self._embed_cols:
            input_list_X.append(X[c].map(self._val_map).fillna(0).values)
        # add rest of columns 
        if len(self._non_embed_cols) > 0:
            input_list_X.append(X[self._non_embed_cols].values)
        return input_list_X