from HybridCORELS import *

# Load data
X, y, features, prediction = load_from_csv("./data/compas_mined.csv")

# Define a hybrid model that will use the Pre-Black-Box training paradigm
# We want a transparency greater than 80%
# We set max_card to 1 as rule mining is already done in our dataset
hyb_model = HybridCORELSPreClassifier(min_coverage=0.8, max_card=1)

# Trains it
hyb_model.fit(X, y, features=features, prediction_name=prediction, time_limit=60, memory_limit=4000)

# Displays it
print(hyb_model)