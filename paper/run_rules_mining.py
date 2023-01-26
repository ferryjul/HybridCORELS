from exp_utils import get_data

datasets = ["compas", "adult", "acs_employ"]

for dataset_name in datasets: # for each dataset, call the get_data method which will mine it if not already available as mined
    X, y, features, prediction = get_data(dataset_name, {"train" : 0.8, "test" : 0.2})
