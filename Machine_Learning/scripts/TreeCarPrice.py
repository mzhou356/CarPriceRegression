import graphviz
from sklearn import tree
from CarPrice import *

class treeCarPrice(CarPrice):
    def __init__(self,data, regressor,oneTree):
        super().__init__(data,regressor)
        self.oneTree = oneTree
        
     
    @property
    def calculateCoef(self):
        model = self.trained_model 
        return model.feature_importances_
    
    @property
    def tree_plot(self):
        """
        This function outputs decision tree plot 
    
        args:
        model: decision tree or random forest 
        features_names: a list of feature names.
        """
        if self.oneTree:
            model = self.trained_model
            if self._trimmed:
                feature_names = self._trimmed_features.columns
            else:
                feature_names = self._features.columns
            dot_data = tree.export_graphviz(model,feature_names=feature_names,
                                    rounded=True)
            graph = graphviz.Source(dot_data,format="png")
            return graph
        else:
            print("This function requires only one tree")