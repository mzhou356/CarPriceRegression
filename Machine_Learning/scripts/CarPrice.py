### This creates a carPrice Class for machine learning 
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


class CarPrice:
    def __init__(self,data, regressor):
        self._features = data.drop("price",axis=1)
        self._trimmed = False
        self._trimmed_features = None
        self._label = data.price 
        self._base = regressor 
        self._trained_model = None
        self._tuned=False
        self._search_result = None 
        
    @property      
    def reset_trimmed(self):
        self._trimmed = False
       
    @property
    def trained_model(self):
        return self._trained_model 
    
    def reset_trained_model(self, model):
        if model == None:
            print ("model is not valid")
        self._trained_model = model
      
        
    def data_split(self,seed,test_size):
        if self._trimmed:
            features = self._trimmed_features
        else:
            features = self._features
        X_train, X_test, y_train, y_test = train_test_split(features,self._label, 
                                                                test_size = test_size, random_state=seed)
        return X_train,X_test,y_train,y_test 
    
    def train_model(self,X,y):
        if self._tuned:
            search_result = copy.deepcopy(self._search_result)
            model = search_result.best_estimator_
        else:
            model = self._base 
        model.fit(X,y)
        self._trained_model = model 

    @property
    def search_result(self):
        return self._search_result
    
    def param_search(self,params,X,y,V=1):
        searchGrid = GridSearchCV(self._base,params,scoring = "neg_root_mean_squared_error",n_jobs=22, verbose=V)
        searchGrid.fit(X,y)
        self._tuned = True
        self._search_result = searchGrid
       
        
    def calculate_pred(self,x,y,retrain):
        if retrain:
            self.train_model(x,y)
        model = self._trained_model
        return model.predict(x)
    
      
    def regression_metrics(self,x_train,y_train,x_test,y_test,retrain=True):
        """
        This function outputs model scores (r2 and root mean squared error)
    
        args:
        x_train: train features, pandas dataframe
        y_train: train label, pandas series
        x_test: test features, pandas dataframe
        y_test: test label, pandas series
        retrain: boolean, retrain model or use trained model 
        
        returns:
        r2 score for both train and test 
        root mean squared error for both train and test 
        as a pandas dataframe
        """     
        pred_train = self.calculate_pred(x_train,y_train,retrain)
        pred_test = self.calculate_pred(x_test,y_test,retrain = False)
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
    
    @property
    def calculateCoef(self):
        model = self._trained_model
        return model.coef_
    
    def linear_feature_importance(self,plot=True):
        """
        This function creates model feature importance for linear regression
        
        returns:
        feature importance pandas dataframe and a bar plot
        """
        coefs = self.calculateCoef;
        if self._trimmed:
            features = self._trimmed_features
        else:
            features = self._features
        table = pd.DataFrame({"features":features.columns,"score":np.abs(coefs)})
        if plot:
            table.sort_values("score",ascending=False).head(20).plot.barh(x="features",y="score",figsize=(6,8),label="coef")
            plt.title("top 20 features")
            plt.legend(loc="top right")
            plt.show()
            table.sort_values("score").head(20).plot.barh(x="features",y="score",figsize=(6,8),label="coef")
            plt.title("bottom 20 features")
            plt.legend(loc="lower right")
            plt.show()
        return table.sort_values("score") 
        
        
    def removeFeatures(self,n):
        """
        This function removed n number of features. 
        
        Arg:
        n: an int, number of features. 
        
        """
        feature_table = self.linear_feature_importance(plot=False)
        features_remove = feature_table.head(n).features 
        self._trimmed_features = self._features.drop(features_remove,axis=1)
        self._trimmed = True
        
    @property
    def price_diff(self):
        """
        This function outputs a dataframe with price diff info 
    
        returns:
        a dataframe with price difference and feature information. 
        """
        model = self._trained_model
        if self._trimmed:
            result_table = self._trimmed_features.copy()
        else:
            result_table = self._features.copy()
        pred_price = self.calculate_pred(result_table,self._label,retrain=False)
        diff = (pred_price-self._label)/self._label*100
        result_table["price_diff_pct"]=diff
        result_table["price_diff_abs"]=np.abs(diff)
        return result_table.sort_values("price_diff_abs",ascending=False)
    
    @property
    def plot_pred_price(self):
        """
        This funciton plots predicted price vs actual price, with a r2 score 
        Also plots residual value distribution 
        """
        if self._trimmed:
            features = self._trimmed_features
        else:
            features = self._features
        model = self._trimmed_features
        y = self._label
        pred = self.calculate_pred(features,y,retrain=False)
        r2 = r2_score(y,pred)
        sns.jointplot(y,pred,label=f"r2_score:{r2}",kind="reg")
        plt.xlabel("price")
        plt.ylabel("predicted price")
        plt.legend(loc="best")
        plt.show()
        sns.distplot((pred-y))
        plt.xlabel("error(pred-price)")
        plt.show()