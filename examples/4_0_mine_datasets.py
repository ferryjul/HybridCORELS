import argparse
from local_config import ccanada_expes
import os
import argparse
from exp_utils import get_data, to_df

rseed = 0
parser = argparse.ArgumentParser()
# train data, last column is label
parser.add_argument("--dataset", type=int, help='Dataset ID', default=0)
#parser.add_argument("--bbox_type", type=str, help='Black box. Options: random_forest, ada_boost, gradient_boost', default='random_forest')
#parser.add_argument("--rseed", type=int, help='Random Data Split', default=0)
#parser.add_argument("--time_limit", type=int, help='Maximum run-time in seconds', default=3600)

args = parser.parse_args()

datasets = ["compas", "adult", "acs_employ"]
dataset_name = datasets[args.dataset]

# Get the data
print("Loading the data for dataset ", dataset_name)
X, y, features, prediction = get_data(dataset_name, {"train" : 0.6, "valid" : 0.20, "test" : 0.20}, 
                                random_state_param=rseed)
df_X = to_df(X, features)
print("Data loaded for dataset ", dataset_name)