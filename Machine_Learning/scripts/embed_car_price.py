# pylint: disable=arguments-differ, too-many-arguments, too-many-locals
"""
This is categorical embed network car price regressor.
"""
from nn_car_price import np, tf, tfk, tfkl, NnCarPrice

class EmbedCarPrice(NnCarPrice):
    """
    This class is for categorical embed of car price regressor.
    """
    @classmethod
    def embed_model_setup(cls, embed_cols, non_embed_cols, x_train,
                          dense_size, dense_output_size,
                          dropout, metrics, lr, embed_size_multiplier=1.0):
        """
        This function sets up models, one embed layer for each categorical feature and
        merge with other models

        Args:
        embed_cols: a list of string, feature name for embeded columns
        x_train: pandas df, features for training
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

        for col in embed_cols:
            c_emb_name = col + "_embedding"
            num_unique = x_train[col].nunique() + 1 # allow null value for label 0
        # use a formula from fastai
            embed_size = int(min(50, num_unique/2*embed_size_multiplier))
            input_model = tfkl.Input(shape=(1,)) # one categorical features at a time for embed
        # each input category gets an embed feature vector
            output_model = tfkl.Embedding(num_unique, embed_size, name=c_emb_name)(input_model)
        # reshape embed model so each corresponding row gets its own feature vector
            output_model = tfkl.Reshape(target_shape=(embed_size,))(output_model)
        # adding all categorical inputs
            input_models.append(input_model)
        # append all embeddings
            output_embeddings.append(output_model)
        #  train other features with one NN layer
        if len(non_embed_cols) > 0:
            input_numeric = tfkl.Input(shape=(len(non_embed_cols),))
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
            output = tfkl.Dense(size, kernel_initializer="uniform",
                                activation=tf.nn.leaky_relu)(output)
        # not add drop out for last output layer
            if i < len(dense_output_size)-1:
                output = tfkl.Dropout(dropout[i])(output)
        output = tfkl.Dense(1, activation="linear")(output)
        model = tfk.models.Model(inputs=input_models, outputs=output)
        model.compile(loss="mse", optimizer=tf.optimizers.Adam(learning_rate=lr), metrics=metrics)
        return model

    @property
    def calculate_coef(self):
        """
        This function estimates layer coefficients by estimating coefficients of the weights avg

        Args:
        embed_size: num of embed an int

        Returns:
        An array of coefficients.
        """
        raise NotImplementedError("not available for EmbedCarPrice")

    def linear_feature_importance(self, plot=True):
        """
        This function creates model feature importance for linear regression

        returns:
        feature importance pandas dataframe and a bar plot
        """
        raise NotImplementedError("not availbale for EmbedCarprice")

    def price_diff(self, X, y, x_list):
        """
        This function outputs a dataframe with price diff info

        args:
        input features to caluclate price difference
        label to compare results
        x_list: converted features for cate embed

        returns:
        a dataframe with price difference and feature information.
        """
        result_table = X.copy()
        pred_price = self.calculate_pred(x_list, y, retrain=False)
        diff = (pred_price-y)/y*100
        result_table["price_diff_pct"] = diff
        result_table["price_diff_abs"] = np.abs(diff)
        return result_table.sort_values("price_diff_abs", ascending=False)
