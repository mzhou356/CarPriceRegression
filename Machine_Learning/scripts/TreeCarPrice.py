import graphviz
from sklearn import tree
from CarPrice import *

class treeCarPrice(LinearCarPrice):
    def __init__(self,data, regressor,oneTree):
        super().__init__(data,regressor)
        self.oneTree = oneTree
            
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
        if self.oneTree:
            model = self._trained_model
            dot_data = tree.export_graphviz(model,feature_names=features,
                                    rounded=True)
            graph = graphviz.Source(dot_data,format="png")
            return graph
        else:
            print("This function requires only one tree")