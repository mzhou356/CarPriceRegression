# pylint: disable=too-many-arguments, attribute-defined-outside-init
"""
DataSetUp Class for split dataset
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataSetUp:
    """
    This class takes in features and label and splits data into train and test.
    This class also makes tensorflow dataset and categorical embed feature list.
    """
    def __init__(self, features, label):
        """
        label: a pandas Series, label column
        feature: feature columns without the label column
        val_map: only for categorical embedding, mapping for categorical mapping
        embed_cols: only for categorical embedding, columns for embedding
        non_embed_cols: only for categorical embedding, columns for non embedded columns
        """
        self.__label = label
        self.__features = features
        self.__val_maps = None
        self.__embed_cols = None
        self.__non_emebd_cols = None

    def data_split(self, seed, test_size, dev_set=False, dev_seed=None, dev_size=None):
        """
        seed: random state for data split
        test_size: a float, percentage of data for testing
        dev_set: include dev set or only test and train.
        dev_seed: random seed for split train data into train and dev
        dev_size: a float, percentage of data for dev and train

        returns:
        pandas data frame of train, test or train, test, and dev
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.__features, self.__label, test_size=test_size, random_state=seed)
        if dev_set:
            x_train, x_dev, y_train, y_dev = train_test_split(
                x_train, y_train, test_size=dev_size, random_state=dev_seed)
            return x_train, x_dev, x_test, y_train, y_dev, y_test
        return x_train, x_test, y_train, y_test

    @classmethod
    def make_tensor_dataset(cls, X, y, batch_size):
        """
        This function generates tensorflow train and test dataset for NN.

        args:
        X: a pandas dataframes, features
        y: a pandas series, label
        batch_size: batch size for training

        returns:
        tensforflow dataset
        """
        data_set = tf.data.Dataset.from_tensor_slices(
            (X.values, y.values)).batch(batch_size).prefetch(
                tf.data.experimental.AUTOTUNE).shuffle(X.shape[0])
        return data_set

    def categorical_mapping(self, x_train, embed_cols):
        """
        This function generates categorical map for entity embedding using training dataset

        Args:
        x_train: a pandas dataframe, training dataset for map categories
        embed_cols:a list of feature name for embed columns
        """
        val_maps = {}
        self.__embed_cols = embed_cols
        self.__non_embed_cols = [col for col in x_train.columns if col not in embed_cols]
        for col in embed_cols:
            raw_values = x_train[col].unique()
            val_maps[col] = {}
            for i, _ in enumerate(raw_values):
        # start with zero so fillna with zero shows the category in
        # new dataset is not in any existing categories)
                val_maps[col][raw_values[i]] = i+1
        self.__val_maps = val_maps

    def cate_data_list(self, X):
        """
        This function transforms features using X_train data to match the format
        to train neural network

        Args:
        X_train: pandas df, features for training
        X: pandas df, test or dev data set

        Returns:
        a python list of features appropriate for categorical embedding
        """
        input_list_x = []

        for col in self.__embed_cols:
            input_list_x.append(X[col].map(self.__val_maps[col]).fillna(0).values)
        # add rest of columns
        if len(self.__non_embed_cols) > 0:
            input_list_x.append(X[self.__non_embed_cols].values)
        return input_list_x
