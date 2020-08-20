import matplotlib.pyplot as plt 
import graphviz
import numpy as np
import pandas as pd
import tensorflow
import tensorflow.compat.v2 as tf 
from sklearn import tree
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

tfk = tf.keras;
tfkl=tf.keras.layers


def linear_feature_importance(features,model,tree_model=False):
    """
    This function creates model feature importance for linear regression
    
    args:
    features: cols of features, a list 
    model: linear regression model 
    tree_model: boolean, true or false 
    
    returns:
    feature importance pandas dataframe and a bar plot
    """
    if tree_model:
        coefs = model.feature_importances_
    else:
        coefs = model.coef_
    table = pd.DataFrame({"features":features,"score":np.abs(coefs)})
    table.sort_values("score",ascending=False).head(20).plot.barh(x="features",y="score",figsize=(6,8),label="coef")
    plt.title("top 20 features")
    plt.legend(loc="top right")
    plt.show()
    table.sort_values("score").head(20).plot.barh(x="features",y="score",figsize=(6,8),label="coef")
    plt.title("bottom 20 features")
    plt.legend(loc="lower right")
    plt.show()
    return table.sort_values("score",ascending=False) 

def regression_metrics(model,x_train,y_train,x_test,y_test):
    """
    This function outputs model scores (r2 and root mean squared error)
    
    args:
    model: a trained model 
    x_train: train features, pandas dataframe
    y_train: train label, pandas series
    x_test: test features, pandas dataframe
    y_test: test label, pandas series
    
    returns:
    r2 score for both train and test 
    root mean squared error for both train and test 
    as a pandas dataframe
    """
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    r2_train = r2_score(y_train,pred_train)
    r2_test = r2_score(y_test,pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train,pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
    metric_table = pd.DataFrame({"r2_score":[r2_train,r2_test],"rmse":[rmse_train,rmse_test]},
                                index=["train","test"])
    metric_table["price_diff_abs_max"] = price_diff(model,x_test,y_test)["price_diff_abs"].max()
    return metric_table
   
def price_diff(model,features,label):
    """
    This function outputs a dataframe with price diff info 
    
    args:
    features: dataframe features
    label: label column  
    data: original data
    
    returns:
    a dataframe with price difference and feature information. 
    """
    result_table = features.copy()
    pred_price = model.predict(features)
    diff = (label-pred_price)/label*100
    result_table["price_diff_pct"]=diff
    result_table["price_diff_abs"]=np.abs(diff)
    return result_table

def tree_plot(model,feature_names):
    """
    This function outputs decision tree plot 
    
    args:
    model: decision tree or random forest 
    features_names: a list of feature names.
    """
    dot_data = tree.export_graphviz(model,feature_names=feature_names,
                                    rounded=True)
    graph = graphviz.Source(dot_data,format="png")
    return graph
    
def set_gpu_limit(n):
    """
    This function sets the max num of GPU in G to minimize overuse GPU per session.
    
    args:
    n: a float, num of GPU in G.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*n)]) 
    
def make_tensor_dataset(X,y,batch_size):
    """
    This function generates tensorflow train and test dataset for NN.
    
    args:
    X: a pandas dataframes, features 
    y: a pandas series, label 
    batch_size: training batch size for each shuffle 
    
    returns:
    tensforflow dataset
    """
    data_set = tf.data.Dataset.from_tensor_slices((X.values,
                 y.values)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(X.shape[0])
    return data_set  
    
    
    
    
    
    
    