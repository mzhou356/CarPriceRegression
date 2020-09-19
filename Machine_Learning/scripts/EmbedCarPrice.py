from NNCarPrice import *

class EmbedCarPrice(NNCarPrice):    
    
    def set_base_model(self,model):
        self._base = model
  
    def train_model(self,train_list,y_train,dev_list,y_dev):
        model = self._base
        self._history = model.fit(train_list,y_train, epochs = self._epochs,
                                     shuffle=True,verbose=1,validation_data=(dev_list,y_dev),
                                     callbacks=self._callbacks)
        self._trained_model = model
        
    def calculate_pred(self,x,y,retrain=True):
        if retrain:
            self._train_model
        model = self._trained_model
        return model.predict(x,batch_size=self._batch_size).flatten()

    @classmethod
    def embed_model_setup(cls,embed_cols,non_embed_cols,X_train,dense_size,dense_output_size,
                          dropout,metrics,lr,embed_size_multiplier=1.0):
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
        # reshape embed model so each corresponding row gets its own feature vector 
            output_model = tfkl.Reshape(target_shape=(embed_size,))(output_model)
        # adding all categorical inputs 
            input_models.append(input_model)
        # append all embeddings 
            output_embeddings.append(output_model)
        #  train other features with one NN layer        
        if len(non_embed_cols)>0:
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

    @property
    def calculateCoef(self):
        """
        This function estimates layer coefficients by estimating coefficients of the weights avg
    
        Args:
        embed_size: num of embed an int 
    
        Returns:
        An array of coefficients. 
        """
        return "not available for EmbedCarPrice"
    
        
    def linear_feature_importance(self,plot=True):
        """
        This function creates model feature importance for linear regression
        
        returns:
        feature importance pandas dataframe and a bar plot
        """
        return "not availbale for EmbedCarprice"
    
    @classmethod    
    def param_search(cls,params,partial_setup,train_input_list,y_train,dev_input_list,y_dev,
                     V,xlog=True,ylog=True,optimizer="loss"):
        """
        Plots paramsearch for one epoch only
    
        Args:
        params: a list of parameters 
        partial_setup: output of model_setup func, partial function 
        xlog,ylog: scale for the graph 
        """
        metrics = []
        for p in params:
            hist = partial_setup(p).fit(train_input_list,y_train, epochs=1,shuffle=True,verbose = V,
                              validation_data=(dev_input_list,y_dev))
            metric = hist.history[optimizer][0]
            metrics.append(metric)
        plt.plot(params,metrics)
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.show()      