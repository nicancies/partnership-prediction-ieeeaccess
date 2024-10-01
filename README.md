This repository relates to the PhD thesis of Mr. Nicola Drago titled "Explainable identification of predictive factors for successful inter-agency partnerships".

The repository "factor-clusterization-visualization" contains the script to embed the factor list through TF-IDF or sBERT, to cluster the factor clusters with K-means, and to perform a bi-gram analysis. It also contains the factor list, and the clustering graphs.

The repository "factor-classification-ml-models" contains the script to process a corpus of project evaluation reports in PDF format, a list of 4-grade project ratings, and the factor list into four machine learning models (logistic regressions, random forests, eXtreme Gradient Boosting, and multilayered perceptron neural networks) trained to predict project rating, SHapley Additive exPlanations (SHAP) values to rank the factors by importance in rating calculations, and boostrap sampling confidence intervals of SHAP values.
