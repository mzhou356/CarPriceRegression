### This creates a carPrice Class for machine learning 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

class carPrice():
    def __init__(self,data, regressor, NN = False, tree = False,batch_size = None):
        self.features = data.drop("price",axis=1)
        self.trimmmed_features = None
        self.label = data.price 
        self.base = regressor 
        self.regressor = regressor 
        self.NN = NN
        self.Tree = tree
        self.tuned=False
        self.gridResult = None 
        self.batch_size = None 
        
        
    def removeFeatures(self, n ):
        feature_table = self.linear_feature_importance()
        features_remove = feature_table.head(n).features 
        self.trimmed_features = self.features.drop(features_remove,axis=1)
        
    def data_split(self,seed,test_size,trimmed = False):
        if trimmed == True:
            features = self.trimmed_features 
        else:
            features = self.features
        X_train, X_test, y_train, y_test = train_test_split(features,self.label, test_size = test_size, random_state=seed)
        return X_train,X_test,y_train,y_test 
    
    def train_model(self,X,y):
        if self.tuned ==True:
            model = self.gridResult.best_estimator_
        else:
            model = self.base 
        model.fit(X,y)
        self.regressor = model 
  
    def regression_metrics(self,x_train,y_train,x_test,y_test):
        """
        This function outputs model scores (r2 and root mean squared error)
    
        args:
        x_train: train features, pandas dataframe
        y_train: train label, pandas series
        x_test: test features, pandas dataframe
        y_test: test label, pandas series
        batch_size: batch_size for predicting NN models
        returns:
        r2 score for both train and test 
        root mean squared error for both train and test 
        as a pandas dataframe
        """
        self.train_model(x_train,y_train)
        model = self.regressor 
        if self.batch_size != None:
            pred_train = model.predict(x_train,batch_size=self.batch_size).flatten()
            pred_test = model.predict(x_test,batch_size=self.batch_size).flatten()
        else:
            pred_train = model.predict(x_train)
            pred_test = model.predict(x_test)
        r2_train = r2_score(y_train,pred_train)
        r2_test = r2_score(y_test,pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train,pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
        metric_table = pd.DataFrame({"r2_score":[r2_train,r2_test],"rmse":[rmse_train,rmse_test]},
                                index=["train","test"])
        metric_table["price_diff_abs_max"] = [np.max(np.abs((y_train-pred_train)/y_train*100)),
                                          np.max(np.abs((y_test-pred_test)/y_test*100))
                                         ]
                                          
        return metric_table
    
    def paramSearch(self,params,X,y):
        searchGrid = GridSearchCV(self.base,params,scoring = "neg_root_mean_squared_error",n_jobs=22, verbose=1)
        searchGrid.fit(X,y)
        self.tuned = True
        self.gridResult = searchGrid
        
    def linear_feature_importance(self):
        """
        This function creates model feature importance for linear regression
    
        args:
        features: cols of features, a list 
        model: linear regression model 
        tree_model: boolean, true or false 
        NN_weights: estimated NN weights. 
    
        returns:
        feature importance pandas dataframe and a bar plot
        """
        model = self.regressor 
        if self.NN:
            coefs = NN_weights 
        elif self.Tree:
            coefs = model.feature_importances_
        else:
            coefs = model.coef_
        table = pd.DataFrame({"features":self.features.columns,"score":np.abs(coefs)})
        table.sort_values("score",ascending=False).head(20).plot.barh(x="features",y="score",figsize=(6,8),label="coef")
        plt.title("top 20 features")
        plt.legend(loc="top right")
        plt.show()
        table.sort_values("score").head(20).plot.barh(x="features",y="score",figsize=(6,8),label="coef")
        plt.title("bottom 20 features")
        plt.legend(loc="lower right")
        plt.show()
        return table.sort_values("score") 