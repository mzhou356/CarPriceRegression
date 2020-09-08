### This creates a carPrice Class for machine learning 
import pandas as pd
import numpy as np
import tensorflow
import tensorflow.compat.v2 as tf 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)



tfk = tf.keras;
tfkl=tf.keras.layers

class carPrice():
    def __init__(self,data, regressor, NN = False, tree = False,batch_size = None):
        self.features = data.drop("price",axis=1)
        self.trimmed_features = None
        self.label = data.price 
        self.base = regressor 
        self.regressor = regressor 
        self.NN = NN
        self.embed = False
        self.Tree = tree
        self.tuned=False
        self.gridResult = None 
        self.batch_size = batch_size
        self.history = None 
        
        
    def removeFeatures(self, n ):
        feature_table = self.linear_feature_importance(plot=False)
        features_remove = feature_table.head(n).features 
        self.trimmed_features = self.features.drop(features_remove,axis=1)
        
    def data_split(self,seed,test_size,trimmed = False):
        if trimmed == True:
            features = self.trimmed_features 
        else:
            features = self.features
        X_train, X_test, y_train, y_test = train_test_split(features,self.label, test_size = test_size, random_state=seed)
        return X_train,X_test,y_train,y_test 
    
    def train_model(self,X=None,y=None,train_dataset = None,dev_dataset=None,epochs=None, V = 0, callbacks=None):
        if self.tuned ==True:
            model = self.gridResult.best_estimator_
        else:
            model = self.base 
        if self.NN:
            self.history = model.fit(train_dataset, epochs = epochs,
                                     shuffle=True,verbose=V,validation_data=dev_dataset,
                                     callbacks=callbacks)
        else:
            model.fit(X,y)
        self.regressor = model 
  
    def regression_metrics(self,x_train,y_train,x_test,y_test,retrain=True,train_dataset=None,dev_dataset=None,epochs=None,
                          V=0,callbacks=None):
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
        if retrain:
            if self.NN:
                self.train_model(train_dataset=train_dataset,dev_dataset=dev_dataset,epochs=epochs,V=V, callbacks=callbacks)
            else:
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
        
    def linear_feature_importance(self,plot=True,NN_coefs = None):
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
            coefs = NN_coefs
        elif self.Tree:
            coefs = model.feature_importances_
        else:
            coefs = model.coef_
        table = pd.DataFrame({"features":self.features.columns,"score":np.abs(coefs)})
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
    
    def price_diff(self,embed_input_list=None,embed_y=None,trimmed=False):
        """
        This function outputs a dataframe with price diff info 
    
        args:
        features: dataframe features
        label: label column  
        data: original data
        cate: if categorical embed default is 0 
        features_input: for categorical embed model 
    
        returns:
        a dataframe with price difference and feature information. 
        """
        model = self.regressor
        label = self.label
        if trimmed:
            result_table = self.trimmed_features.copy()
        else:
            result_table = self.features.copy()
        if self.NN:
            if self.embed:
                label = embed_y
                pred_price = model.predict(embed_input_list, batch_size=self.batch_size).flatten()
            else:
                pred_price = model.predict(result_table.values, batch_size=self.batch_size).flatten()
        else:
            pred_price = model.predict(result_table)
        diff = (pred_price-label)/label*100
        result_table["price_diff_pct"]=diff
        result_table["price_diff_abs"]=np.abs(diff)
        return result_table.sort_values("price_diff_abs",ascending=False)
    
    def plot_pred_price(self,X_embed=None,y_embed=None,trimmed=False):
        """
        This funciton plots predicted price vs actual price, with a r2 score 
        Also plots residual value distribution 
    
        Args:
        model: trained machine learning model 
        X: features, pandas dataframe or numpy array
        y: label, numpy array or pandas Series
        batch_size: if has a value, it is NN mdl
        """
        if trimmed:
            features = self.trimmed_features
        else:
            features = self.features
        model = self.regressor
        y = self.label
        if self.NN:
            if self.embed:
                y = y_embed
                pred = model.predict(X_embed, batch_size=self.batch_size).flatten()
            else:
                pred = model.predict(features.values,batch_size=self.batch_size).flatten()
        else:
            pred = model.predict(features)
        r2 = r2_score(y,pred)
        sns.jointplot(y,pred,label=f"r2_score:{r2}",kind="reg")
        plt.xlabel("price")
        plt.ylabel("predicted price")
        plt.legend(loc="best")
        plt.show()
        sns.distplot((pred-y))
        plt.xlabel("error(pred-price)")
        plt.show()