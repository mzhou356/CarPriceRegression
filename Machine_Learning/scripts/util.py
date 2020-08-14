import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV


def linear_feature_importance(features,model):
    """
    This function creates model feature importance for linear regression
    
    args:
    features: cols of features, a list 
    model: linear regression model 
    
    returns:
    feature importance pandas dataframe and a bar plot
    """
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