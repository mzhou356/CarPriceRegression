#pylint: disable=too-many-arguments
"""
This creates a carPrice linear Class for machine learning
"""
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

class CarPriceLinear:
    """
    This is the linear regression class for car price predictor.
    """
    def __init__(self, regressor):
        """
        trimed_features: if feature elimination is performed, then it will have a
        pandas dataframe of features
        base: base regressor during initialization
        trained_model: tuned model
        search_result: if gridSearch is performed, it will have search result
        """
        self.__base = regressor
        self.__trained_model = None
        self.__search_result = None

    def reset_trained_model(self, model):
        """
        resets trained model for the linear regressor instance.
        """
        if model is None:
            print("model is not valid")
        self.__trained_model = model

    def train_model(self, X, y):
        """
        This function trains the base regressor

        Args:
        X: features
        y: label
        """
        if self.__search_result:
            search_result = copy.deepcopy(self.__search_result)
            model = search_result.best_estimator_
        else:
            model = self.__base
        model.fit(X, y)
        self.__trained_model = model

    def save_model(self, file_path):
        """
        saves trained model.
        """
        joblib.dump(self.__trained_model, file_path)

    @classmethod
    def load_model(cls, file_path):
        """
        loads model based upon file_path.
        """
        return joblib.load(file_path)

    @property
    def search_result(self):
        """
        getter for gridsearchCV results.
        """
        return self.__search_result

    def param_search(self, params, X, y, verbose=1, metrics="neg_root_mean_squared_error",
                     worker_num=22):
        """
        perform gridsearchCV to get best params
        """
        search_grid = GridSearchCV(self.__base, params, scoring=metrics,
                                   n_jobs=worker_num, verbose=verbose, refit=False)
        search_grid.fit(X, y)
        self.__search_result = search_grid

    def calculate_pred(self, X, y, retrain):
        """
        predicts label using trained model.
        """
        if retrain:
            self.train_model(X, y)
        model = self.__trained_model
        return model.predict(X)

    def regression_metrics(self, X, y, ind, retrain=True):
        """
        This function outputs model scores (r2 and root mean squared error)

        args:
        X: features, pandas dataframe
        y: label, pandas series
        ind: a string, index for the metrics, "train","test","dev"
        retrain: boolean, retrain model or use trained model

        returns:
        r2 score and root mean squared error for data_set, price different abs max %
        as a pandas dataframe
        """
        pred = self.calculate_pred(X, y, retrain)
        r_2 = r2_score(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        metric_table = pd.DataFrame({"r2_score":[r_2], "rmse":[rmse]},
                                    index=[ind])
        metric_table["price_diff_abs_max"] = [np.max(np.abs((y-pred)/y*100))]
        return metric_table

    @property
    def calculate_coef(self):
        """
        extracts feature importance of the model.
        """
        model = self.__trained_model
        return model.coef_

    def linear_feature_importance(self, features, plot=True):
        """
        This function creates model feature importance for linear regression

        Args:
        features: a dataframe with features only

        returns:
        feature importance pandas dataframe and a bar plot
        """
        coefs = self.calculate_coef
        table = pd.DataFrame({"features":features.columns, "score":np.abs(coefs)})
        if plot:
            table.sort_values(
                "score", ascending=False).head(20).plot.barh(
                    x="features", y="score", figsize=(6, 8), label="coef")
            plt.title("top 20 features")
            plt.legend(loc="top right")
            plt.show()
            table.sort_values("score").head(20).plot.barh(
                x="features", y="score", figsize=(6, 8), label="coef")
            plt.title("bottom 20 features")
            plt.legend(loc="lower right")
            plt.show()
        return table.sort_values("score")

    def remove_features(self, features, num):
        """
        This function removed n number of features.

        Arg:
        features: orgininal features before trimming
        num: an int, number of features.

        Returns:
        trimmed features as pandas data frame
        """
        feature_table = self.linear_feature_importance(features, plot=False)
        features_remove = feature_table.head(num).features
        trimmed_features = features.drop(features_remove, axis=1)
        return trimmed_features

    def price_diff(self, X, y):
        """
        This function outputs a dataframe with price diff info

        args:
        input features to caluclate price difference
        label to compare results

        returns:
        a dataframe with price difference and feature information.
        """
        result_table = X.copy()
        pred_price = self.calculate_pred(result_table, y, retrain=False)
        diff = (pred_price - y)/y*100
        result_table["price_diff_pct"] = diff
        result_table["price_diff_abs"] = np.abs(diff)
        return result_table.sort_values("price_diff_abs", ascending=False)

    def plot_pred_price(self, X, y, retrain=False):
        """
        This funciton plots predicted price vs actual price, with a r2 score
        Also plots residual value distribution

        args:
        input features to caluclate price difference
        label to compare results
        retrain: retrain model on the entire feature set
        """
        pred = self.calculate_pred(X, y, retrain=retrain)
        r_2 = r2_score(y, pred)
        sns.jointplot(y, pred, label=f"r2_score:{r_2}", kind="reg")
        plt.xlabel("price")
        plt.ylabel("predicted price")
        plt.legend(loc="best")
        plt.show()
        sns.distplot((pred - y))
        plt.xlabel("error(pred-price)")
        plt.show()
