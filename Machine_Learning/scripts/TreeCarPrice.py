import graphviz
from sklearn import tree
from LinearCarPrice import *

class treeCarPrice(CarPriceLinear):
    def __init__(self,regressor,oneTree):
        super().__init__(regressor)
        self._oneTree = oneTree
            
    @property
    def calculateCoef(self):
        model = self._trained_model 
        return model.feature_importances_
    
    def tree_plot(self,features):
        """
        This function outputs decision tree plot 
    
        args:
        features: a list of strings, input features column names 
        """
        if self._oneTree:
            model = self._trained_model
            dot_data = tree.export_graphviz(model,feature_names=features,
                                    rounded=True)
            graph = graphviz.Source(dot_data,format="png")
            return graph
        else:
            raise NotImplementedError("This function requires only one tree")