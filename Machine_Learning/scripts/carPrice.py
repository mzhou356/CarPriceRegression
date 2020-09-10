### This creates a carPrice Class for machine learning 
import pandas as pd
import graphviz
import numpy as np
import tensorflow
import tensorflow.compat.v2 as tf 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)



tfk = tf.keras;
tfkl=tf.keras.layers

class carPrice:
    def __init__(self,data, regressor):
        self.features = data.drop("price",axis=1)
        self.trimmed_features = None
        self.label = data.price 
        self.base = regressor 
        self.regressor = regressor 
        self.tuned=False
        self.gridResult = None 
         
    def removeFeatures(self, n ):
        feature_table = self.linear_feature_importance(plot=False)
        features_remove = feature_table.head(n).features 
        self.trimmed_features = self.features.drop(features_remove,axis=1)
        
    def data_split(self,seed,test_size,trimmed = False):
        if trimmed == True:
            X_train, X_test, y_train, y_test = train_test_split(self.trimmed_features,self.label, 
                                                                test_size = test_size, random_state=seed) 
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.features,self.label, 
                                                                test_size = test_size, random_state=seed)
        return X_train,X_test,y_train,y_test 
    
    def train_model(self,X,y):
        if self.tuned:
            model = self.gridResult.best_estimator_
        else:
            model = self.base 
        model.fit(X,y)
#         if self.NN:
#             if self.embed:
#                 train_input,y_train = train_dataset
#                 self.history = model.fit(train_input,y_train,epochs=epochs,
#                                          shuffle=True,verbose=V,validation_data=dev_dataset,
#                                          callbacks=callbacks)
#             else:
#                 self.history = model.fit(train_dataset, epochs = epochs,
#                                      shuffle=True,verbose=V,validation_data=dev_dataset,
#                                      callbacks=callbacks)
#         else:   
        self.regressor = model 
    
    def calculate_pred(self,x,y,retrain=True):
        if retrain:
            self.train_model(x,y)
        model = self.regressor 
        return model.predict(x)
  
    def regression_metrics(self,x_train,y_train,x_test,y_test,retrain=True):
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
        pred_train = self.calculate_pred(x_train,y_train,retrain)
        pred_test = self.calculate_pred(x_test,y_test,retrain)
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
    
    def paramSearch(self,params,X,y,V=1):
        searchGrid = GridSearchCV(self.base,params,scoring = "neg_root_mean_squared_error",n_jobs=22, verbose=V)
        searchGrid.fit(X,y)
        self.tuned = True
        self.gridResult = searchGrid
        
    def calculateCoef(self):
        model = self.regressor 
        return model.coef_
        
    def linear_feature_importance(self,plot=True):
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
        coefs = self.calculateCoef();
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
    
    def price_diff(self,trimmed=False):
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
#         if self.NN:
#             if self.embed:
#                 label = embed_y
#                 pred_price = model.predict(embed_input_list, batch_size=self.batch_size).flatten()
#             else:
#                 pred_price = model.predict(result_table.values, batch_size=self.batch_size).flatten()
        pred_price = self.calculate_pred(result_table,label,retrain=False)
        diff = (pred_price-label)/label*100
        result_table["price_diff_pct"]=diff
        result_table["price_diff_abs"]=np.abs(diff)
        return result_table.sort_values("price_diff_abs",ascending=False)
    
    def plot_pred_price(self,trimmed=False):
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
#         if self.NN:
#             if self.embed:
#                 y = y_embed
#                 pred = model.predict(X_embed, batch_size=self.batch_size).flatten()
#             else:
#                 pred = model.predict(features.values,batch_size=self.batch_size).flatten()
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
        
class treeCarPrice(carPrice):
    def calculateCoef(self):
        model = self.regressor 
        return model.feature_importances_
    
    def tree_plot(self,trimmed=False):
        """
        This function outputs decision tree plot 
    
        args:
        model: decision tree or random forest 
        features_names: a list of feature names.
        """
        model = self.regressor
        if trimmed:
            feature_names = self.trimmed_features
        else:
            feature_names = self.features
        dot_data = tree.export_graphviz(model,feature_names=feature_names,
                                    rounded=True)
        graph = graphviz.Source(dot_data,format="png")
        return graph
    
class NNCarPrice(carPrice):
    def __init__(self,data,NN_model=None,batch_size,epochs,callbacks,layers):
        super().__init(data,NN_mode)
        self.history = None;
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.layers = layers
        
    def data_split(self,seed,test_size,trimmed = False):
        X_train,X_test,y_train,y_test = super().data_split(seed,test_size,trimmed)
        X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,
                                                       test_size = test_size,random_state=seed)
        return X_train,X_test,y_train,y_test,X_dev,y_dev
    
        
    def train_model(self,train_dataset,dev_dataset,V):
        model = self.base 
        self.history = model.fit(train_dataset, epochs = self.epochs,
                                     shuffle=True,verbose=V,validation_data=dev_dataset,
                                     callbacks=self.callbacks)
        self.regressor = model
        
    def calculate_pred(self,x,train_dataset,dev_dataset,V,retrain=True):
        if retrain:
            self.train_model(train_dataset,dev_dataset=dev_dataset,V=V)
        model = self.regressor 
        return model.predict(x,batch_size=self.batch_size).flatten()
    
    @staticmethod
    def set_gpu_limit(n):
        """
        This function sets the max num of GPU in G to minimize overuse GPU per session.
    
        args:
        n: a float, num of GPU in G.
        """
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*n)]) 
        
    def make_tensor_dataset(self,X,y)
        """
        This function generates tensorflow train and test dataset for NN.
    
        args:
        X: a pandas dataframes, features 
        y: a pandas series, label 
      
    
        returns:
        tensforflow dataset
        """
        data_set = tf.data.Dataset.from_tensor_slices((X.values,
                 y.values)).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(X.shape[0])
        return data_set  
    
    def make_model(self,sizes,input_size, metrics, l2 = 10e-5, lr = 1e-4):
        """
        This function creates tensorflow regression model and use l2 to regularize the 
        activate output. 
    
        Args:
        sizes: a list of integer indicating num of hidden nodes
        input_size: num of input features, an integer
        l2:option input for activation regularizer to prevent overfitting of training data 
        lr: for optimizer adam, gradient descent step size 
        metrics: metrics for optimizing the model during tuning 
        """
        layers = [tfkl.InputLayer(input_shape=input_size)]
        for s in sizes:
            layers.append(tfkl.Dense(units=s,activation=tf.nn.leaky_relu,activity_regularizer=tfk.regularizers.l2(l2)))
        layers.append(tfkl.Dense(units=1))
        model = tfk.Sequential(layers, name = "NN_regressor")
        model.compile(optimizer = tf.optimizers.Adam(learning_rate=lr), 
                  loss = "mse",
                  metrics = metrics)
        self.base = model
        
    def extract_weights(self,layer_num):
        """
        This function returns model weights for the specific model layer num
        
        Args:
        model: NN trained model. 
        layer_num: layer_num, an integer.
        
        Returns:
        layer weights as numpy array 
        """
        model = self.regressor
        weights_info = model.layers[layer_num]
        return weights_info.weights[0].numpy()
    
     def calculateCoef(self):
        """
        This function estimates layer coefficients by estimating coefficients of the weights avg
    
        Args:
        model: NN trained model. 
        layers: layers of model 
    
        Returns:
        An array of coefficients. 
        """
        mean_weights = extract_weights(0)
        layers = self.layers
        for i in range(1,layers):
            weights = extract_weights(i)
            mean_weights = np.dot(mean_weights,weights)
        return mean_weights
    
    def plot_metrics(self,metric):
        """
        This function plots the metric value for train and test 
    
        Arg:
        history: a tensorflow.python.keras.callbacks.History object 
        metric: a string, type of metric to plot 
    
        Returns:
        A plot that shows 2 overlapping loss versus epoch images. red is for test and blue is for train 
    
        """
        history = self.history.history
        plt.plot(history[metric], color="blue", label="train")
        plt.plot(history[f"val_{metric}"], color="red", label="test")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title("model training results")
        plt.legend(loc="best")
        plt.show()
        
    def paramSearch(self,params,mdl_setup,train_data_set,dev_dataset,V,xlog=True,ylog=True,metrics="loss"):
        """
        Plots paramsearch for one epoch only
    
        Args:
        params: a list of parameters 
        mdl_setup: output of model_setup func, partial function 
        xlog,ylog: scale for the graph 
        """
        metrics = []
        for p in params:
            hist = mdl_setup(p).fit(train_data_set, epochs=1,shuffle=True,verbose = V,
                              validation_data=dev_dataset)
        metric = hist.history[metrics][0]
        metrics.append(metric)
        plt.plot(params,metrics)
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.show()
        
def embedCarPrice(NNCarPrice):
    def cate_embed_process(self,X_train,X_dev,X_test,embed_cols):
        """
        This function transforms features using X_train data to match the format to train neural network 
    
        Args:
        X_train: pandas df, features for training 
        X_dev: pandas df, features for dev/validation
        X_test: pandas df, features for test (hold out set)
        embed_cols: a list of feature name for embeded columns 
    
        Returns:
        a list of features for train, dev, and test 
        """
        input_list_train = []
        input_list_dev = []
        input_list_test = []
    
        for c in embed_cols:
            raw_values = X_train[c].unique() # get num of unique categories for each categorical features 
            val_map={} # map each categorical to an integer number for embedding 
            for i in range(len(raw_values)):
                val_map[raw_values[i]] = i+1 
        # map all categories to a value based upon train value only 
            input_list_train.append(X_train[c].map(val_map).values)
            input_list_dev.append(X_dev[c].map(val_map).fillna(0).values)  # allow null value as its own categorical (class 0 for null)
            input_list_test.append(X_test[c].map(val_map).fillna(0).values)
        # add rest of columns 
        other_cols = [c for c in X_train.columns if c not in embed_cols]
        if len(other_cols) > 0:
            input_list_train.append(X_train[other_cols].values)
            input_list_dev.append(X_dev[other_cols].values)
            input_list_test.append(X_test[other_cols].values)
        return input_list_train,input_list_dev,input_list_test
    
    def embed_model_setup(self,embed_cols,X_train,dense_size,dense_output_size,dropout,metrics,lr,embed_size_multiplier=1.0):
        """
        This function sets up models, one embed layer for each categorical feature and merge with other models 
    
        Args:
        embed_cols: a list of string, feature name for embeded columns 
        X_train: pandas df, features for training 
        numeric_types: a list of types for each column 
        dense_size: a list of input hidden node size for numeric feature model 
        dense_output_size: a list of output hidden node size after all embed layers added together 
        droput ratio: for drop out layer, a list 
        metrics: metrics for optimizing models 
        lr: learning rate for adam optimizer 
        embed_size_multiplier: float, adjusts embedsize 
    
        Returns:
        Embeded neural network model 
        """
        input_models = [] 
        output_embeddings = [] 
    
        for c in embed_cols:
            c_emb_name = c+"_embedding"
            num_unique = X_train[c].nunique()+1 # allow null value for label 0 
        # use a formula from fastai
            embed_size = int(min(50,num_unique/2*embed_size_multiplier))
            input_model = tfkl.Input(shape=(1,)) # one categorical features at a time for embed 
        # each input category gets an embed feature vector 
            output_model = tfkl.Embedding(num_unique, embed_size, name = c_emb_name)(input_model) 
#         print(output_model.shape)
        # reshape embed model so each corresponding row gets its own feature vector 
            output_model = tfkl.Reshape(target_shape=(embed_size,))(output_model)
#         print(output_model.shape)
        
        # adding all categorical inputs 
            input_models.append(input_model)
        
        # append all embeddings 
            output_embeddings.append(output_model)
        
    #  train other features with one NN layer 
        numeric_cols = [c for c in X_train.columns if c not in embed_cols]
        if len(numeric_cols)>0:
            input_numeric = tfkl.Input(shape=(len(numeric_cols),))
            for i, size in enumerate(dense_size):
                if i == 0:
                    embed_numeric = tfkl.Dense(size)(input_numeric)
                else:
                    embed_numeric = tfkl.Dense(size)(embed_numeric)
            input_models.append(input_numeric)
            output_embeddings.append(embed_numeric)
    
    # add everything together at the end 
    # add all output embedding nodes together as one layer 
        output = tfkl.Concatenate()(output_embeddings)
        for i, size in enumerate(dense_output_size):
            output = tfkl.Dense(size,kernel_initializer="uniform",activation = tf.nn.leaky_relu)(output)
        # not add drop out for last output layer 
            if i < len(dense_output_size)-1:
                output = tfkl.Dropout(dropout[i])(output)
        output = tfkl.Dense(1,activation="linear")(output)
    
        model = tfk.models.Model(inputs = input_models, outputs = output)
        model.compile(loss="mse",optimizer=tf.optimizers.Adam(learning_rate=lr),metrics=metrics)
        self.base = model