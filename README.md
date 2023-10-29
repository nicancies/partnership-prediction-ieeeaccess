# partnership-prediction
Predictive factors for interagency partnership success

The files contained in this repository are part of a PhD research project aimed at identifying and classifying factors that affect success or failure of partnerships between multilateral agencies.
Factor classification by criticality aims at increasing multilateral agencies' capacity to ex-ante identify high-potential partnerships, enhancing partnership effectiveness and lowering partnership appraisal costs.
This research project is divided in three parts: (factor) identification, visualization, and classification. More details on the first two parts of this project maay be found in a paper that was submitted on October 19th, 2023 to a Q1 peer-reviewed open-access journal - more details will follow. The third part of this research project is ongoing.
The L1 classification of factors is based on the Author's manual detection of emphasis differences between critical factors vs. non-critical ones in individual systematic evaluations of partnerships and other works on partnerships in literature.
The third part of this research project will focus on higher level classifications.

The repository contains the following files:
1) A polished dataset with 750+ factors, associated to their bibliographic source and an initial classification (L1) by criticality. 
2) Python code to cluster and visualize the dataset.
3) 2D and 3D scatterplots visualizing the clustered dataset by feature reduction level. The feature reduction is based on semantic similarity (cosine similarity ratio) between the embeddings of the above-mentioned dataset.
4) 2D and 3D scatterplots visualizing the L1 (first level) classification of such factors by criticality, distinguishing L1-critical factors in color from non-critical factors in grey.
5) 2D and 3D scatterplots visualizing the sole L1-critical factors.
6) bar-charts depicting TF-IDF analyses on bigrams of different authors of systematic partnership evaluations.
7) an elbow diagram helping to determine best DBscan coefficients for factor clusering trials.

More files may follow.
