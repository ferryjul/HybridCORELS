## Python Module for Learning Hybrid Interpretable (Rule-Based) & Black Box models - Using the "Interpr then BB training" paradigm

Python module for Hybrid Rule-List/Black-Box models. This implementation contains methods for the two hybrid models learning paradigms:

* `HybridCORELSPreClassifier` uses the "Interpr then BB training" paradigm: the interpretable part of the Hybrid model is trained first, using a modified version of the CORELS algorithm. Given a desired minimum transparency level, the algorithm returns the certifiaby optimal prefix. Then, a BB model is trained to perform well on the remaining examples. The BB can then be specialized on such examples. In this case, optimality is certified for the interpretable part alone, which is guaranteed to have the best objective function given the coverage constraint.

*  `HybridCORELSPostClassifier` uses the "BB then Interpr training" paradigm: the BB part of the Hybrid model is trained first. Then, the interpretable part is trained, using a modified version of the CORELS algorithm. Given a desired minimum transparency level, along with the BB predictions, the algorithm returns the certifiaby optimal prefix. Because it is trained after the BB, the interpretable part is able to correct its mistakes. In this case, optimality is certified for the entire model, which is guaranteed to have the best objective function given the coverage constraint and the black-box mistakes.

<p align = "center"><img src = "./example_HybridCORELS_model_MLP_2.png"></p><p align = "center">
Example Hybrid Model learnt on the COMPAS dataset using HybridCORELS along with a standard sklearn MLPClassifier as black-box model.
</p>
