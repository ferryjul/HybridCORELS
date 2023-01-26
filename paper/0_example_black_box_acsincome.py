import os 
import numpy as np 

random_state_value = 42
test_size_ratio = 0.3

# Load the new Adult dataset (using the provided folktables module)
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["TX"], download=True)
X, y, group = ACSIncome.df_to_numpy(acs_data)
print("Dataset shape: ", X.shape)
#X = X[0:10000]
#y = y[0:10000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=random_state_value)

# Create and train the hyperparameter-optimized black-box model
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe

os.environ["HYPEROPT_FMIN_SEED"] = str(random_state_value) # for reproductibility

black_box = HyperoptEstimator(classifier=any_classifier("optimized_black_box"),
                            refit=True, # to refit on the entire training set (default)
                            preprocessing= [], # any_preprocessing("optimized_preprocessing") to search all preprocessings OR [] for no preprocessing
                            n_jobs=1, # (default)
                            verbose=False, #False,
                            algo=tpe.suggest,
                            max_evals=100,
                            seed=random_state_value, # for reproducibility
                            trial_timeout=60) #HyperoptEstimator()

black_box.fit(X_train, y_train, cv_shuffle=True, valid_size=0.3, random_state=np.random.default_rng(random_state_value*2))

print("Chosen black-box model: ", black_box._best_learner)
print("Chosen preproc: ", black_box._best_preprocs)

# Evaluate it

train_acc = black_box.score(X_train, y_train)
print("Training accuracy = %.3f" %train_acc)

test_acc = black_box.score(X_test, y_test)
print("Test accuracy = %.3f" %test_acc)