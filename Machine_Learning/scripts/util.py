import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV


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
    return table 

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
    
    
    
    
    
    
    
    
    