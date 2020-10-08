"""
1. This script reads input from user as a json object.
2. Processes the data into feature format for prediction.
"""
import json
from util import load_dict, input_process

# load in json object
with open("user_input.json") as fobj:
    user_input = json.load(fobj)

# load in maps for feature processing 
# numeric binning maps 
year_map = load_dict("year_map.pkl")
mileage_map = load_dict("mil_dict.pkl")
mpg_map = load_dict("mpg_dict.pkl")
engine_map = load_dict("engine_dict.pkl")
binning_maps = [year_map,mileage_map,mpg_map,engine_map]

# categorical embed mapping 
cate_map = load_dict("saved_models/cate_map.pkl")

# transform the numeric column into binned features
binned_columns = input_process(user_input, binning_maps)


