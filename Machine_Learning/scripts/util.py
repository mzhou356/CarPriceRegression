import matplotlib.pyplot as plt 
import seaborn as sns
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


def linear_feature_importance(features,model,tree_model=False,NN_weights=None):
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
    if tree_model:
        coefs = model.feature_importances_
    elif NN_weights is not None:
        coefs = NN_weights
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

def regression_metrics(model,x_train,y_train,x_test,y_test,batch_size=None):
    """
    This function outputs model scores (r2 and root mean squared error)
    
    args:
    model: a trained model 
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
    if batch_size:
        pred_train = model.predict(x_train,batch_size=batch_size).flatten()
        pred_test = model.predict(x_test,batch_size=batch_size).flatten()
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
   
def price_diff(model,features,label,batch_size=None):
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
    if batch_size:
        pred_price = model.predict(features.values ,batch_size=batch_size).flatten()
    else:
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
    

def make_model(sizes,input_size, metrics, l2 = 10e-5, lr = 1e-4):
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
    return model   

def plot_metrics(history,metric):
    """
    This function plots the metric value for train and test 
    
    Arg:
    history: a tensorflow.python.keras.callbacks.History object 
    metric: a string, type of metric to plot 
    
    Returns:
    A plot that shows 2 overlapping loss versus epoch images. red is for test and blue is for train 
    
    """
    history = history.history
    plt.plot(history[metric], color="blue", label="train")
    plt.plot(history[f"val_{metric}"], color="red", label="test")
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title("model training results")
    plt.legend(loc="best")
    plt.show()
    
def plot_pred_price(model,X,y,batch_size=None):
    """
    This funciton plots predicted price vs actual price, with a r2 score 
    Also plots residual value distribution 
    
    Args:
    model: trained machine learning model 
    X: features, pandas dataframe or numpy array
    y: label, numpy array or pandas Series
    batch_size: if has a value, it is NN mdl
    """
    if batch_size:
        pred = model.predict(X, batch_size=batch_size).flatten()
    else:
        pred = model.predict(X)
    r2 = r2_score(y,pred)
    sns.jointplot(y,pred,label=f"r2_score:{r2}",kind="reg")
    plt.xlabel("price")
    plt.ylabel("predicted price")
    plt.legend(loc="best")
    plt.show()
    sns.distplot((pred-y))
    plt.xlabel("error(pred-price)")
    plt.show()
    
    
def extract_weights(model,layer_num):
    """
    This function returns model weights for the specific model layer num
        
    Args:
    model: NN trained model. 
    layer_num: layer_num, an integer.
        
    Returns:
    layer weights as numpy array 
    """
    weights_info = model.layers[layer_num]
    return weights_info.weights[0].numpy()

def coeff_estimation(model,layers):
    """
    This function estimates layer coefficients by estimating coefficients of the weights avg
    
    Args:
    model: NN trained model. 
    layers: layers of model 
    
    Returns:
    An array of coefficients. 
    """
    mean_weights = extract_weights(model,0)
    for i in range(1,layers):
        weights = extract_weights(model,i)
        mean_weights = np.dot(mean_weights,weights)
    return mean_weights

def cate_embed_process(X_train,X_dev,X_test,embed_cols):
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
    input_list_train.append(X_train[other_cols].values)
    input_list_dev.append(X_dev[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    return input_list_train,input_list_dev,input_list_test

def embed_model_setup(embed_cols,X_train,dense_size,dense_output_size,dropout,metrics,lr):
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
    
    Returns:
    Embeded neural network model 
    """
    input_models = [] 
    output_embeddings = [] 
    
    for c in embed_cols:
        c_emb_name = c+"_embedding"
        num_unique = X_train[c].nunique()+1 # allow null value for label 0 
        # use a formula from Jeremy Howard
        embed_size = int(min(600,round(1.6*np.power(num_unique,0.56))))
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
    input_numeric = tfkl.Input(shape=(len([c for c in X_train.columns if c not in embed_cols]),))
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
    return model 
     
    