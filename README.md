- Check out the application [here](https://mindy-dossett.com/2020/10/26/Car-Price-App/
)

# CarPriceRegression
  - Predict the price of a car using various regression models.
  
### Data set:
  * The detailed information and the data sets can be found in this [kaggle link](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes/tasks?taskId=1258).
  
### Folders:  
  * data_process: contains notebooks and utils.py to convert raw car price csv files into machine learning ready data:
    * scripts:
      * [utils.py](https://github.com/mzhou356/CarPriceRegression/blob/master/data_process/scripts/utils.py): contains helper functions for car_price_data_merging_cleaning notebook.
    * notebooks:
      * [car_price_data_merging_cleaning.ipynb](https://github.com/mzhou356/CarPriceRegression/blob/master/data_process/notebooks/car_price_data_merging_cleaning.ipynb): contains detailed information on data import and feature engineering. 
      * [categorical_feature_for_linear_regressin.ipynb](https://github.com/mzhou356/CarPriceRegression/blob/master/data_process/notebooks/categorical_feature_for_linear_regression.ipynb): contains information on converting categorical features for various machine learning regression models.
  * machine_learning: contains scripts and notebooks to train car price regression models:
    * scripts:
       * [data_set_up.py](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/scripts/data_set_up.py): DataSetUp class that processed data into train, dev, and test. It also creates tensorflow datasets and categorical embedding data sets. 
       * [linear_car_price.py](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/scripts/linear_car_price.py): CarPriceLinear class for training and tuning elasticnet linear regression models. Parent class for all other models classes.
       * [tree_car_price.py](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/scripts/tree_car_price.py): TreeCarPrice class for training and tuning tree based models. 
       * [nn_car_price.py](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/scripts/nn_car_price.py): NnCarPrice class for training and tuning fully connected neural net regression models without dropout layers and categorical embeddings. 
       * [embed_car_price.py](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/scripts/embed_car_price.py): EmbedCarPrice class for training and tuning neural net regression models with dropout layers and categorical embeddings. 
     * notebooks:
       * [linear_regression.ipynb](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/notebooks/linear_regression.ipynb): notebook for elastic net linear regression models. 
       * [tree_regressor.ipynb](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/notebooks/tree_regressor.ipynb): notebook for tree based models: decision tree, random forest, and xgboost. 
       * [neural_network_regression.ipynb](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/notebooks/tree_regressor.ipynb): notebook for neural network regression models without entity embedding. 
       * [neural_network_with_categorical_embedding.ipynb](https://github.com/mzhou356/CarPriceRegression/blob/master/machine_learning/notebooks/neural_network_with_categorical_embedding.ipynb): notebook for neural network models with categorical embedding. 
       
 ### Conclusions:
   * Both decision tree and random forest did better than neural network regression models without entity embedding. It could be due to small training data sets and/or tree based algorithms work great for this type of applications. 
   * Neural network with entity embedding worked slightly better than tree based models in terms of maximum price difference but not R2 score. A bigger dataset may provide more insights with neural network approach. 
   * All models have a difficult time predicting prices for cars that tend to be older especially if the older cars have limited mileage. RF models and Neural network models tend to perform slightly better on different cars. An ensemble approach with weighted average will be used for a car price model application project. 
 
