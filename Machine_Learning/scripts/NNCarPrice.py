import tensorflow
import tensorflow.compat.v2 as tf 
from CarPrice import *
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

tfk = tf.keras;
tfkl=tf.keras.layers

class NNCarPrice(CarPrice):
    def __init__(self,data,NN_model,batch_size,epochs,callbacks):
        super().__init__(data,NN_model)
        self._history = None;
        self._batch_size = batch_size
        self._epochs = epochs
        self._callbacks = callbacks
        self._layers = len(NN_model.layers)
        self._train_dataset = None
        self._dev_dataset = None
        
    def data_split(self,seed,test_size):
        X_train,X_test,y_train,y_test = super().data_split(seed,test_size)
        X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,
                                                       test_size = test_size,random_state=seed)
        return X_train,X_test,y_train,y_test,X_dev,y_dev
    
    @property
    def train_model(self):
        model = self._base
        self._history = model.fit(self._train_dataset, epochs = self._epochs,
                                     shuffle=True,verbose=1,validation_data=self._dev_dataset,
                                     callbacks=self._callbacks)
        self._trained_model = model
        
    def calculate_pred(self,x,y,retrain=True):
        if retrain:
            self.train_model
        model = self.trained_model
        return model.predict(x.values,batch_size=self._batch_size).flatten()
    
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
  
    def make_tensor_dataset(self,X_train,y_train,X_dev,y_dev):
        """
        This function generates tensorflow train and test dataset for NN.
    
        args:
        X: a pandas dataframes, features 
        y: a pandas series, label 
      
    
        returns:
        tensforflow dataset
        """
        train_set = tf.data.Dataset.from_tensor_slices((X_train.values,
                 y_train.values)).batch(self._batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(X_train.shape[0])
        dev_set = tf.data.Dataset.from_tensor_slices((X_dev.values,
                 y_dev.values)).batch(self._batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(X_dev.shape[0])
        self._train_dataset = train_set
        self._dev_dataset = dev_set
        
    @classmethod
    def make_model(cls,sizes,input_size, metrics, l2, lr):
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
        
    def extract_weights(self,layer_num):
        """
        This function returns model weights for the specific model layer num
        
        Args:
        model: NN trained model. 
        layer_num: layer_num, an integer.
        
        Returns:
        layer weights as numpy array 
        """
        model = self.trained_model
        weights_info = model.layers[layer_num]
        return weights_info.weights[0].numpy()
    
    @property
    def calculateCoef(self):
        """
        This function estimates layer coefficients by estimating coefficients of the weights avg
    
        Args:
        model: NN trained model. 
        layers: layers of model 
    
        Returns:
        An array of coefficients. 
        """
        mean_weights = self.extract_weights(0)
        layers = self._layers
        for i in range(1,layers):
            weights = self.extract_weights(i)
            mean_weights = np.dot(mean_weights,weights)
        return mean_weights.flatten()
    
    def plot_metrics(self,metric):
        """
        This function plots the metric value for train and test 
    
        Arg:
        history: a tensorflow.python.keras.callbacks.History object 
        metric: a string, type of metric to plot 
    
        Returns:
        A plot that shows 2 overlapping loss versus epoch images. red is for test and blue is for train 
    
        """
        history = self._history.history
        plt.plot(history[metric], color="blue", label="train")
        plt.plot(history[f"val_{metric}"], color="red", label="test")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title("model training results")
        plt.legend(loc="best")
        plt.show()
        
    def param_search(self,params,partial_setup,V,xlog=True,ylog=True,optimizer="loss"):
        """
        Plots paramsearch for one epoch only
    
        Args:
        params: a list of parameters 
        partial_setup: output of model_setup func, partial function 
        xlog,ylog: scale for the graph 
        """
        metrics = []
        for p in params:
            hist = partial_setup(p).fit(self._train_dataset, epochs=1,shuffle=True,verbose = V,
                              validation_data=self._dev_dataset)
            metric = hist.history[optimizer][0]
            metrics.append(metric)
        plt.plot(params,metrics)
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.show()
        
    @classmethod    
    def save_model(cls,model,filepath):
        model.save(filepath)
    @classmethod    
    def load_model(cls,filepath):
        return tfk.models.load_model(filepath,custom_objects={"leaky_relu":tf.nn.leaky_relu})