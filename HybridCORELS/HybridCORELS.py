from .corels import PrefixCorelsClassifier
from .utils import check_consistent_length, check_array, get_feature, check_in, check_features
import numpy as np

class HybridCORELSClassifier:
    """Hybrid Rule List/Black-box based classifier.

    This class implements an Hybrid interpretable/black-box model.
    It uses a modified version of the CORELS algorithm (and its Python binding PyCORELS) to learn the intepretable part.
    This "interpretable part" consists in a prefix (ordered set of rules).
    Examples not determined by the prefix's rules are not classifier by a default prediction (as in traditional rule lists),
    but are rather used to train a black-box model, whose base class is user-defined.

    Attributes
    ----------
    c, n_iter, map_type, policy, verbosity, ablation, max_card, min_support : arguments of the CORELS algorithm 
    (see CORELS' documentation for details)

    black_box_classifier: 

    alpha: black-box specialization coefficient (used to weight the black-box training set)

    verbosity: as in original CORELS, + "hybrid" to print information regarding the Hybrid model learning framework

    References
    ----------
    Original CORELS algorithm: Elaine Angelino, Nicholas Larus-Stone, Daniel Alabi, Margo Seltzer, and Cynthia Rudin.
    Learning Certifiably Optimal Rule Lists for Categorical Data. KDD 2017.
    Journal of Machine Learning Research, 2018; 19: 1-77. arXiv:1704.01701, 2017
    """
    
    _estimator_type = "classifier"

    def __init__(self, black_box_classifier=None, c=0.01, n_iter=10000, map_type="prefix", policy="lower_bound",
                 verbosity=["rulelist"], ablation=0, max_card=2, min_support=0.01, beta=0.0, alpha=0.0, min_coverage=0.0, random_state=42):
        # Retrieve parameters related to CORELS, and creation of the interpretable part of the Hybrid model
        self.c = c
        self.n_iter = n_iter
        self.map_type = map_type
        self.policy = policy
        self.verbosity = verbosity
        self.ablation = ablation
        self.max_card = max_card
        self.min_support = min_support
        self.beta = beta
        self.alpha = alpha
        self.min_coverage=min_coverage
        self.interpretable_part = PrefixCorelsClassifier(self.c, self.n_iter, self.map_type, self.policy, self.verbosity, self.ablation, self.max_card, self.min_support, self.beta, self.min_coverage)
        np.random.seed(random_state);
        # Creation of the black-box part of the Hybrid model
        if black_box_classifier is None:
            print("Unspecified black_box_classifier parameter, using sklearn MLPClassifier() for black-box part of the model.")
            from sklearn.neural_network import MLPClassifier
            black_box_classifier = MLPClassifier()
        self.BlackBoxClassifier = black_box_classifier
        self.black_box_part = self.BlackBoxClassifier

        # Done!
        self.is_fitted = False
        if "hybrid" in self.verbosity:
            print("Hybrid model created!")

    def fit(self, X, y, features=[], prediction_name="prediction", specialization_auto_tuning=False):
        """
        Build a CORELS classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8.

        y : array-line, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        features : list, optional(default=[])
            A list of strings of length n_features. Specifies the names of each
            of the features. If an empty list is provided, the feature names
            are set to the default of ["feature1", "feature2"... ].

        prediction_name : string, optional(default="prediction")
            The name of the feature that is being predicted.

        Returns
        -------
        self : obj
        """
        # 1) Fit the interpretable part of the Hybrid model
        if "hybrid" in self.verbosity:
            print("Fitting the interpretable part...")
        self.interpretable_part.fit(X, y, features, prediction_name)

        # 2) Fit the black-box part of the Hybrid model (using examples not determined by the interpretable part)
        # Retrieve only examples not captured by the interpretable part
        interpretable_predictions = self.interpretable_part.predict(X)
        not_captured_indices = np.where(interpretable_predictions == 2)
        captured_indices = np.where(interpretable_predictions < 2)
        if "hybrid" in self.verbosity:
            print("Interpretable part coverage = ", (y.size-not_captured_indices[0].size)/y.size)
            print("Interpretable part accuracy = ", np.mean(interpretable_predictions[captured_indices] == y[captured_indices]))
            print("Fitting the black-box part on examples not captured by the interpretable part...")

        # Old way: fit black-box only on uncaptured examples only
        X_not_captured = X[not_captured_indices]
        y_not_captured = y[not_captured_indices]
        #self.black_box_part.fit(X_not_captured, y_not_captured)

        if not specialization_auto_tuning:
            # New way: weight training examples
            # First compute and set the sample weights
            examples_weights = np.ones(y.shape) # start from the uniform distribution
            examples_weights[not_captured_indices] = np.exp(self.alpha)
            examples_weights /= np.sum(examples_weights)

            # Then train the black-box on its weighted training set
            self.black_box_part.fit(X, y, sample_weight=examples_weights)
        else:
            print("Not implemented yet")
        '''else:
            alpha_list = np.arange(0,15,1)
            np.random.randint(y.shape, replace='False')
            for alpha_value in alpha_list:
                # First compute and set the sample weights
                examples_weights = np.ones(y.shape) # start from the uniform distribution
                examples_weights[not_captured_indices] = np.exp(self.alpha)
                examples_weights /= np.sum(examples_weights)

                # Then train the black-box on its weighted training set
                self.black_box_part.fit(X, y, sample_weight=examples_weights)'''
                
        # Finally set the black-box metrics
        self.black_box_support = not_captured_indices[0].size # Proportion of training examples falling into the black-box part
        if not_captured_indices[0].size > 0:
            self.black_box_accuracy = self.black_box_part.score(X_not_captured, y_not_captured) # Black-Box accuracy on these examples
            y_not_captured_unique_counts = np.unique(y_not_captured, return_counts=True)[1]
            self.black_box_majority = max(y_not_captured_unique_counts)/sum(y_not_captured_unique_counts)
            if "hybrid" in self.verbosity:
                print("majority pred = ", self.black_box_majority)
        else:
            self.black_box_accuracy = 1.00            
        # Done!
        self.is_fitted = True

        return self

    def predict(self, X):
        """
        Predict classifications of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications of the input samples.
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict(X)
        overall_predictions = interpretable_predictions
        
        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict(X)          
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]

        # Return overall prediction
        return overall_predictions

    def predict_proba(self, X):
        """
        Predict classification probabilities of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples, 2].
            The classifications probabilities of the input samples.
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict_proba(X)
        overall_predictions = interpretable_predictions
        
        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict_proba(X)
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]
        
        # Return overall prediction
        return overall_predictions

    def predict_with_type(self, X):
        """
        Predict classifications of the input samples X, along with a boolean (one per example)
        indicating whether the example was classified by the interpretable part of the model or not.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p, t : array of shape = [n_samples], array of shape = [n_samples].
            p: The classifications of the input samples
            t: The part of the Hybrid model which decided for the classification (1: interpretable part, 0: black-box part).
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict(X)
        overall_predictions = interpretable_predictions
        # Craft the predictions type vector
        predictions_type = np.ones(shape=overall_predictions.shape)

        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict(X)        
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]    
            predictions_type[not_captured_indices] = 0

         # Return overall prediction along with the part that classified the example (1: interpretable part, 0: black-box)
        return overall_predictions, predictions_type

    def __str__(self):
        s = "HybridCORELSClassifier"

        if self.is_fitted:
            s += "\n" + self.interpretable_part.rl().__str__()
            s += "\n    default: " + str(self.black_box_part) + "(support %d, accuracy %.3f (majority pred %.3f))" %(self.black_box_support, self.black_box_accuracy, self.black_box_majority)
        else:
            s += "Not Fitted Yet!"
            
        return s

    def score(self, X, y):
        """
        Score the algorithm on the input samples X with the labels y. Alternatively,
        score the predictions X against the labels y (where X has been generated by 
        `predict` or something similar).

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
            The input samples, or the sample predictions. All features must be binary.
        
        y : array-like, shape = [n_samples]
            The input labels. All labels must be binary.

        Returns
        -------
        a : float
            The accuracy, from 0.0 to 1.0, of the rulelist predictions
        """

        labels = check_array(y, ndim=1)
        p = check_array(X)
        check_consistent_length(p, labels)
        
        if p.ndim == 2:
            p = self.predict(p)
        elif p.ndim != 1:
            raise ValueError("Input samples must have only 1 or 2 dimensions, got " + str(p.ndim) +
                             " dimensions")

        a = np.mean(np.invert(np.logical_xor(p, labels)))

        return a