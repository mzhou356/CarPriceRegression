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
