This repository relates to the paper "Explainable Predictive Factors for Inter-agency Partnership Success" under review at IEEE Access.
The paper is based on the PhD research project of Mr. Nicola Drago (University of Padova, Italy) titled "Explainable identification of predictive factors for successful inter-agency partnerships." The thesis of this project was submitted for revision on September 30th, 2024, its final version was approved on December 3rd, 2024, and its defense is scheduled for March 13th, 2025.

The repository contains three folders:

1. "factor-clusterization-visualization" provides the script to comprehend (clustering, 3D visualization) a list of 753 factors that influence inter-agency performances. The script uses TF-IDF or sBERT embeddings, clusters with K-means, reduces dimensionality with principal component analysis, and performs a bi-gram analysis. The repository also contains the factor list and the clustering graphs.

2. "corpus-scripts" provides the scripts for downloading, filtering, and re-naming a corpus of project evaluation and design reports in PDF format. These open documents constitute the evidence input for classifying, by criticality, the key factors for predicting inter-agency partnership performance. The scripts point to co-financed and non-co-financed projects of the Asian Development Bank (AsDB), the International Fund for Agricultural Development (IFAD), the United Nations Development Program (UNDP), and the World Bank (WB).

3. "factor-classification-ml-models" provides the script to process the corpus of project evaluation reports in PDF format, a list of project ratings re-classified on a four-grade scale, and the 753-factor list into four machine learning models (logistic regressions, random forests, eXtreme Gradient Boosting, and multilayered perceptron neural networks). The models undergo supervised learning to classify the projects by rating based on project evaluation report contents. SHapley Additive exPlanations (SHAP) values rank the factors by importance in rating calculations. Bootstrap sampling calculates SHAP value confidence intervals. The comparison between the rate of correct project classification with the 753-factor set and factor subsets shows the predictive capacity of the latter.
