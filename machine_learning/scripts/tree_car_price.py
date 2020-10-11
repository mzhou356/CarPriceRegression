"""
This is child class of CarPriceLinear class.
This class is for tree based models.
"""
import graphviz
from sklearn import tree
from linear_car_price import CarPriceLinear

class TreeCarPrice(CarPriceLinear):
    """
    This is tree based model regressor class.
    """
    def __init__(self, regressor, one_tree):
        """
        Initialize method for tree based models.

        Args:
        regressor: base tree model (decision tree, random forest, xgboost).
        one_tree: Boolean. Is it one tree model or many tree models.
        """
        super().__init__(regressor)
        self.__one_tree = one_tree

    @property
    def __calculate_coef(self):
        """
        This extracts feature importance for tree based models.
        """
        model = self.__trained_model
        return model.feature_importances_

    def tree_plot(self, features):
        """
        This function outputs decision tree plot

        args:
        features: a list of strings, input features column names
        """
        if not self.__one_tree:
            raise NotImplementedError("This function requires only one tree")
        model = self.__trained_model
        dot_data = tree.export_graphviz(model, feature_names=features,
                                        rounded=True)
        graph = graphviz.Source(dot_data, format="png")
        return graph
     