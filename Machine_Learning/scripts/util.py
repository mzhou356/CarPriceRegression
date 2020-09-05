import matplotlib.pyplot as plt 
import seaborn as sns
from functools import partial 
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
    if len(other_cols) > 0:
        input_list_train.append(X_train[other_cols].values)
        input_list_dev.append(X_dev[other_cols].values)
        input_list_test.append(X_test[other_cols].values)
    return input_list_train,input_list_dev,input_list_test

def embed_model_setup(embed_cols,X_train,dense_size,dense_output_size,dropout,metrics,lr,embed_size_multiplier=1.0):
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
    return model 
     
def param_search(params,mdl_setup,train_input_c,y_train_c,dev_input_c,y_dev_c,xlog=True,ylog=True):
    """
    Plots learning rate over loss for lr search
    
    Args:
    lr_rates: a list of learning rates 
    mdl_setup: output of model_setup func, partial function 
    xlog,ylog: scale for the graph 
    """
    losses = []
    for p in params:
        hist = mdl_setup(p).fit(train_input_c,y_train_c, epochs=1,shuffle=True,verbose = 1, 
                              validation_data=(dev_input_c,y_dev_c))
        loss = hist.history["loss"][0]
        losses.append(loss)
    plt.plot(params,losses)
    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")
    plt.show()
    