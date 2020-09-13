from NNCarPrice import *

class EmbedCarPrice(NNCarPrice):    
    def __init__(self,data,NN_model,batch_size,epochs,callbacks):
        super().__init__(data,NN_model,batch_size,epochs,callbacks)
        self._feature_list = None
        self._train_list = None
        self._dev_list = None
        self._test_list =None
        
    @property   
    def train_model(self):
        model = self._base
        self._history = model.fit(self._train_list[0],self._train_list[1], epochs = self._epochs,
                                     shuffle=True,verbose=1,validation_data=self._dev_list,
                                     callbacks=self._callbacks)
        self._trained_model = model
        
    def calculate_pred(self,x,y,retrain=True):
        if retrain:
            self.train_model
        model = self.trained_model
        return model.predict(x,batch_size=self._batch_size).flatten()

    def cate_embed_process(self,X_train,y_train,X_dev,y_dev,X_test,y_test,embed_cols):
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
        input_feature = []
    
        for c in embed_cols:
            raw_values = X_train[c].unique() # get num of unique categories for each categorical features 
            val_map={} # map each categorical to an integer number for embedding 
            for i in range(len(raw_values)):
                val_map[raw_values[i]] = i+1 
            # map all categories to a value based upon train value only 
            input_list_train.append(X_train[c].map(val_map).values)
            # allow null value as its own categorical (class 0 for null)
            input_list_dev.append(X_dev[c].map(val_map).fillna(0).values)  
            input_list_test.append(X_test[c].map(val_map).fillna(0).values)
            input_feature.append(self._features[c].map(val_map).fillna(0).values)
        # add rest of columns 
        other_cols = [c for c in X_train.columns if c not in embed_cols]
        if len(other_cols) > 0:
            input_list_train.append(X_train[other_cols].values)
            input_list_dev.append(X_dev[other_cols].values)
            input_list_test.append(X_test[other_cols].values)
            input_feature.append(self._features[other_cols].values)
        self._feature_list = input_feature
        self._train_list = (input_list_train,y_train)
        self._dev_list = (input_list_dev,y_dev)
        self.test_list = input_list_test
        
    @classmethod
    def embed_model_setup(cls,embed_cols,X_train,dense_size,dense_output_size,dropout,metrics,lr,embed_size_multiplier=1.0):
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
        # print(output_model.shape)
        # reshape embed model so each corresponding row gets its own feature vector 
            output_model = tfkl.Reshape(target_shape=(embed_size,))(output_model)
        # print(output_model.shape)
        
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
    
            
    @property
    def price_diff(self):
        """
        This function outputs a dataframe with price diff info 
    
        returns:
        a dataframe with price difference and feature information. 
        """
        model = self.trained_model
        if self._trimmed:
            result_table = self._trimmed_features.copy()
        else:
            result_table = self._features.copy()
        pred_price = self.calculate_pred(self._feature_list,self._label,retrain=False)
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
        y = self._label
        pred = self.calculate_pred(self._feature_list,y,retrain=False)
        r2 = r2_score(y,pred)
        sns.jointplot(y,pred,label=f"r2_score:{r2}",kind="reg")
        plt.xlabel("price")
        plt.ylabel("predicted price")
        plt.legend(loc="best")
        plt.show()
        sns.distplot((pred-y))
        plt.xlabel("error(pred-price)")
        plt.show()