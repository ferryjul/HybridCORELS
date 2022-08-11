## Python Module for Learning Hybrid Interpretable (Rule-Based) & Black Box models - Using the 

Python module for Hybrid Rule-List/Black-Box models. This implementation uses the "Interpr then BB training" paradigm: the interpretable part of the Hybrid model is trained first, using a modified version of the CORELS algorithm. Given a desired minimum transparency level, the algorithm returns the certifiaby optimal prefix. Then, a BB model is trained to perform well on the remaining examples. The BB can then be specialized on such examples.


<p align = "center"><img src = "./example_HybridCORELS_model_MLP_2.png"></p><p align = "center">
Example Hybrid Model learnt on the COMPAS dataset using HybridCORELS along with a standard sklearn MLPClassifier as black-box model.
</p>
