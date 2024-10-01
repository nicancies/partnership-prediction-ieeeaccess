#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Python code for the standardization, clusterization and visualization of factors that influence interagency partnerships

# The code is available under GPLv3 license at https://github.com/nicancies/partnership-prediction

# Author: Nicola Drago, Oct. 19th, 2023

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import difflib
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from gsdmm import MovieGroupProcess
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
# from wordcloud import WordCloud


nltk.download('wordnet')

# Load the Excel file
df_literature = pd.read_excel(r'Partnership-factor-list-750.xlsx', sheet_name='Foglio1')

# Get the 'factor' column as a list
l_literature = df_literature['factor'].tolist()

# Tokenization
l_tokenized = [word_tokenize(str(i)) for i in l_literature]

# Remove stopwords
stop_words = stopwords.words('english')
l_stopworded = [[word for word in sentence if word.lower() not in stop_words] for sentence in l_tokenized]

# Remove punctuation
punctuation = set(string.punctuation)
l_depunctuated = [[word for word in sentence if word.lower() not in punctuation] for sentence in l_stopworded]

for (i, item) in enumerate(l_depunctuated, start=1):
    print(i, item)

# Lemmatization - OPTION A, list l_a
lemmatizer = WordNetLemmatizer()
l_lemmatized_a = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in l_depunctuated]
l_lemmatized_joined_a = [' '.join(sentence) for sentence in l_lemmatized_a]

l_a = l_lemmatized_joined_a

for (i, item) in enumerate(l_a, start=1):
    print(i, item)



# In[2]:


# SECTION 2, FEATURE EXTRACTION

# SECTION 2.1, pairs of similar literature-based factors

print("\nPAIRS OF SIMILAR LITERATURE-BASED FACTORS \n")

similarity = [0.50, 0.60, 0.70, 0.8]  # Manually-defined range of similarity coefficients for the difflib.SequenceMatcher

print("Range of coefficients utilized to calculate similarity between factors: \n", similarity, "\n")

# Find similar lists and count the similar records
similar_lists = []
for w in range(len(similarity)):
    for i in range(len(l_a)):
        for j in range(i + 1, len(l_a)):
            if difflib.SequenceMatcher(None, l_a[i], l_a[j]).ratio() > similarity[w]:
                similar_lists.append((i, l_a[i], j, l_a[j]))

    print(f"Similar lists with similarity > {similarity[w]}:")
    print("Length of the list:", len(similar_lists))
    for pair in similar_lists:
        print(pair)
    print()

    # Find the three list records with the largest number of similar records
    counter = Counter([pair[0] for pair in similar_lists])
    top_three = counter.most_common(3)

    for record, count in top_three:
        print("Record:", l_a[record])
        print("Number of similar records:", count)
        print()

    print()



# In[3]:


# SECTION 2, FEATURE EXTRACTION

# SECTION 2.2, removal of pairs of similar literature-based factors


print("DECISION TO TAKE: now that you have the similar pairs of factors from literature, please input the lowest similarity coefficient that does not cut useful factors \n")

decided_similarity = float(input("Enter a numeric input for decided_similarity e.g. the paper used 0.8: "))

print("You entered this value for the similarity coefficient:  ", decided_similarity, "\n")

print("\nSIMPLIFIED LIST OF LITERATURE-BASED FACTORS \n")

l_s = l_a.copy()
to_remove = []
for i in range(len(l_a)):
    for j in range(i + 1, len(l_a)):
        if difflib.SequenceMatcher(None, l_a[i], l_a[j]).ratio() > decided_similarity:
            if len(l_a[i]) <= len(l_a[j]):
                # add the list with the least words to the list of lists to remove
                to_remove.append(l_a[i])
            else:
                to_remove.append(l_a[j])

for list_to_remove in to_remove:
    if list_to_remove in l_s:  # Check if the element exists in the list before removing it
        l_s.remove(list_to_remove)

for index, row in enumerate(l_s):
    print(f"{index}: {row}")


    
# SECTION 2, FEATURE EXTRACTION

# SECTION 2.3, removal of the two most frequent words (may not be done if bigrams or trigrams are used)     
    
    
# Create an instance of TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# # Apply TF-IDF transformation
# tfidf_matrix = tfidf_vectorizer.fit_transform(l_a), now l_a has been turned into l_s
# Apply TF-IDF transformation
tfidf_matrix = tfidf_vectorizer.fit_transform(l_s)  # Change l_a to l_tfidf, then changing to l_s


# Get the feature names (words) in the TF-IDF matrix
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a dictionary to store word-score pairs
word_scores = {}

# Collect the TF-IDF scores for each word
for i in range(len(l_s)):     # chaning l_a into l_s
    for col in tfidf_matrix[i].nonzero()[1]:
        word = feature_names[col]
        tfidf_score = tfidf_matrix[i, col]
        if word in word_scores:
            word_scores[word] += tfidf_score
        else:
            word_scores[word] = tfidf_score

# Sort the words by reverse TF-IDF score
sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

# Get the top two words with the highest TF-IDF scores
top_words = [word for word, _ in sorted_words[:2]]

# Print the explanation for the top words
print(f"The following words were removed from the list because they are so frequent to possibly confuse the GSDMM clustering:\n{', '.join(top_words)}\n")

# Create a new list l_tfidf by removing the top words from l_a, now l_a has been turned to l_s
l_tfidf = []
for i, sentence in enumerate(l_s):
    words = sentence.split()
    filtered_words = [word for word in words if word not in top_words]
    l_tfidf.append(f"{i}: {' '.join(filtered_words)}")

print("Lemmatized and reduced list (l_tfidf) of factors without the two most frequent words in the corpus\n")

# Print the new list l_tfidf
for sentence in l_tfidf:
    print(sentence)


# In[4]:


# SECTION 3, CLUSTERIZATION WITH K-MEANS

from sklearn.cluster import KMeans

Kvalue = int(input("Enter a numeric input for the envisaged maximum no. of clusters to consider by the KMeans (e.g. the paper used 10): "))

# Apply TF-IDF transformation to l_tfidf instead of l_a
tfidf_matrix_tfidf = tfidf_vectorizer.fit_transform(l_tfidf)

kmeans = KMeans(n_clusters=Kvalue, random_state=0).fit(tfidf_matrix_tfidf)

# Add cluster labels to l_tfidf
clusters = kmeans.labels_
l_tfidf_clustered = list(zip(l_tfidf, clusters))

# Print the number of factors per topic
print('Number of factors per topic :', Counter(clusters))


# In[5]:


# SECTION 4, TF-IDF KEYWORDS

from sklearn.feature_extraction.text import TfidfVectorizer

print('\n', 'MOST FREQUENT WORDS BY CLUSTER ACCORDING TO KMeans', '\n')

# Convert the list of factors and their corresponding cluster labels into a DataFrame
df_clustered = pd.DataFrame(l_tfidf_clustered, columns=['Factor', 'Cluster'])

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Create a dictionary to store the top 3 words for each cluster
top_words_by_cluster = {}

# For each cluster, calculate the TF-IDF scores
for cluster in set(df_clustered['Cluster']):
    # Filter the dataframe to only include factors from the current cluster
    df_cluster = df_clustered[df_clustered['Cluster'] == cluster]
    # Calculate the TF-IDF matrix for factors in this cluster
    tfidf_matrix = vectorizer.fit_transform(df_cluster['Factor'])
    # Get the feature names (words) in the TF-IDF matrix
    feature_names = vectorizer.get_feature_names_out()
    # Sum the TF-IDF score for each word
    word_scores = tfidf_matrix.sum(axis=0).A1
    # Create a dictionary to store word-score pairs
    word_score_dict = dict(zip(feature_names, word_scores))
    # Sort the words by reverse TF-IDF score
    sorted_words = sorted(word_score_dict.items(), key=lambda x: x[1], reverse=True)
    # Get the top words with the highest TF-IDF scores
    top_words = sorted_words[:20]
    # Store the top words for this cluster
    top_words_by_cluster[cluster] = top_words

# Print the top 3 words for each cluster
for cluster, top_words in top_words_by_cluster.items():
    print(f'Cluster {cluster}: {top_words}')

# Create a DataFrame to store the top words for each cluster
df_top_words = pd.DataFrame([(cluster, word, score) for cluster, words in top_words_by_cluster.items() for word, score in words], columns=['Cluster', 'Word', 'TF-IDF Score'])

# Save the DataFrame to an Excel file
df_top_words.to_excel(f'top_words_cluster_{Kvalue}.xlsx', index=False)
print(f"Results saved to top_words_cluster_{Kvalue}.xlsx")


# In[6]:


# Convert the list of factors and their corresponding cluster labels into a DataFrame
df_clustered = pd.DataFrame(l_tfidf_clustered, columns=['Factor', 'Cluster'])

# Group the factors by cluster and print each group
for cluster, group in df_clustered.groupby('Cluster'):
    print(f'Cluster {cluster}:')
    print(group['Factor'])
    print('\n')
    
    
# Convert the list of factors and their corresponding cluster labels into a DataFrame
df_clustered = pd.DataFrame(l_tfidf_clustered, columns=['Factor', 'Cluster'])

# Save the DataFrame to an Excel file
df_clustered.to_excel('factor_clusters.xlsx', index=False)

print("Clustered factors saved to 'factor_clusters.xlsx'")


# In[7]:


print(l_tfidf)


# In[8]:


df_literature.to_json('df_literature.jsonl', orient='records', lines=True)


# In[9]:


# VISUALIZATION OF DATA BASED ON A PRINCIPAL COMPONENT ANALYSIS AND TF-IDF

# Import necessary libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Calculate the TF-IDF matrix for the l_tfidf list
tfidf_matrix = vectorizer.fit_transform(l_tfidf)

# Create a PCA object with 3 components
pca = PCA(n_components=3)

# Apply PCA to the TF-IDF matrix
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

# Use the kmeans object from Section 3 to get the cluster labels
clusters = kmeans.labels_

# Custom function to display colorbar with centered cluster numbers
def display_colorbar(ax, scatter_obj):
    # Get unique clusters
    unique_clusters = sorted(list(set(clusters)))
    boundaries = np.arange(min(unique_clusters)-0.5, max(unique_clusters)+1.5)
    cbar = plt.colorbar(scatter_obj, ax=ax, label='Cluster', boundaries=boundaries, ticks=unique_clusters)
    cbar.set_ticklabels([str(int(cluster)) for cluster in unique_clusters])

# 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tfidf_pca[:, 0], tfidf_pca[:, 1], tfidf_pca[:, 2], c=clusters, cmap='tab10')
ax.set_xlabel('PC1: context and operations', labelpad=10)
ax.set_ylabel('PC2: governance and management', labelpad=10)
ax.set_zlabel('PC3: project quality', labelpad=10)
display_colorbar(ax, scatter)

# Adjust the view angle for better visibility
ax.view_init(elev=30, azim=-60)  # You can adjust these values for desired view
plt.tight_layout()
plt.show()

# 2D Visualizations for each pair of principal components

# PC1-PC2 Plane
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tfidf_pca[:, 0], tfidf_pca[:, 1], c=clusters, cmap='tab10')
plt.xlabel('PC1: context and operations')
plt.ylabel('PC2: governance and management')
display_colorbar(plt.gca(), scatter)
plt.title('Projection on PC1-PC2 Plane')
plt.grid(True)
plt.show()

# PC2-PC3 Plane
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tfidf_pca[:, 1], tfidf_pca[:, 2], c=clusters, cmap='tab10')
plt.xlabel('PC2: governance and management')
plt.ylabel('PC3: project quality')
display_colorbar(plt.gca(), scatter)
plt.title('Projection on PC2-PC3 Plane')
plt.grid(True)
plt.show()

# PC3-PC1 Plane
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tfidf_pca[:, 2], tfidf_pca[:, 0], c=clusters, cmap='tab10')
plt.xlabel('PC3: project quality')
plt.ylabel('PC1: context and operations')
display_colorbar(plt.gca(), scatter)
plt.title('Projection on PC3-PC1 Plane')
plt.grid(True)
plt.show()

# 4. For more spherical clusters, it is possible to adjust KMeans parameters like n_init and max_iter. 
# Example: KMeans(n_clusters=Kvalue, random_state=0, n_init=50, max_iter=500).fit(tfidf_matrix)


# In[10]:


# Get the component loadings
loadings = pca.components_

# Get the feature names from the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Display top contributing words for each principal component
top_n = 10  # Number of top contributing words to display for each component
for i, component in enumerate(loadings):
    sorted_idx = component.argsort()[::-1]
    top_features = [feature_names[idx] for idx in sorted_idx[:top_n]]
    print(f"Principal Component {i + 1}:")
    print(", ".join(top_features))
    print("\n")

# This code will display the top 10 words (or features) that contribute the most to each of the first three principal components. You can adjust top_n to show more or fewer top contributing words.


# In[11]:


# GRAPHICAL COMPARISON BETWEEN ALL FACTORS AND CRITICAL FACTORS ONLY, IN 3D AND 2D


import matplotlib.pyplot as plt
import numpy as np
# ... [Other necessary imports]

# Assuming PCA (`tfidf_pca`) and clustering (`clusters`) have been calculated previously...

# Specifying the number of clusters
n_clusters = 10

# Identify indices for "CSF" and "CFF"
csf_cff_indices = df_literature[df_literature['class'].isin(['CSF', 'CFF'])].index
valid_indices = [idx for idx in csf_cff_indices if idx < len(tfidf_pca) and idx < len(clusters)]

# Determine global min and max for consistent plot limits
global_min = np.min(tfidf_pca)
global_max = np.max(tfidf_pca)

def plot_data(pca_data, cluster_data, plot_title):
    # ... [No changes here, keeping previous settings]

    # 3D Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=cluster_data, cmap='tab10', s=50)
    ax.set_xlim([global_min, global_max])
    ax.set_ylim([global_min, global_max])
    ax.set_zlim([global_min, global_max])
    ax.set_xlabel('PCA1: Context and Operations')
    ax.set_ylabel('PCA2: Governance and Management')
    ax.set_zlabel('PCA3: Project Quality')
    ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.view_init(elev=30, azim=-60)
    ax.set_title(f'3D Visualization: {plot_title}')
    plt.show()

    # 2D Scatter Plots
    fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
    scatter_plots = []
    scatter_plots.append(axarr[0].scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_data, cmap='tab10', s=50))
    scatter_plots.append(axarr[1].scatter(pca_data[:, 1], pca_data[:, 2], c=cluster_data, cmap='tab10', s=50))
    scatter_plots.append(axarr[2].scatter(pca_data[:, 2], pca_data[:, 0], c=cluster_data, cmap='tab10', s=50))

    # Set axis labels and limits for all plots
    for i, ax in enumerate(axarr):
        ax.set_xlim([global_min, global_max])
        ax.set_ylim([global_min, global_max])
        ax.set_xlabel(f'PCA{i%3 + 1}')
        ax.set_ylabel(f'PCA{(i+1)%3 + 1}')
        ax.set_title(f'2D Visualization: {plot_title}')

    # Adjust spacing for better visibility
    fig.subplots_adjust(bottom=0.2, hspace=0.4)

    # Display colorbar
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])  # positioning the colorbar below the plots
    cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, orientation='horizontal', ticks=np.linspace(0, n_clusters - 1, n_clusters))
    cbar.set_label('Cluster')

    # Adjust tick labels for better centering
    cbar.set_ticks(np.linspace(0, n_clusters - 1, n_clusters))
    cbar.ax.set_xticklabels(range(n_clusters), ha='center')

    plt.show()

# Plot all factors
plot_data(tfidf_pca, clusters, 'All Factors')

# Plot only "CSF" and "CFF" factors
plot_data(tfidf_pca[valid_indices], clusters[valid_indices], 'CSF and CFF Factors')


# In[12]:


import matplotlib.pyplot as plt
import numpy as np
# ... [Other necessary imports]

# Assuming PCA (`tfidf_pca`) and clustering (`clusters`) have been calculated previously...

# Specifying the number of clusters
n_clusters = 10

# Identify indices for "CSF" and "CFF"
csf_cff_indices = df_literature[df_literature['class'].isin(['CSF', 'CFF'])].index

# Verify that indices are valid for the tfidf_pca and clusters arrays
valid_indices = [idx for idx in csf_cff_indices if idx < len(tfidf_pca) and idx < len(clusters)]

# Create a function to plot the data (to avoid code duplication)
def plot_data(pca_data, cluster_data, plot_title):
    # 3D Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=cluster_data, cmap='tab10', s=50)
    ax.set_xlabel('PCA1: Context and Operations', labelpad=10)
    ax.set_ylabel('PCA2: Governance and Management', labelpad=10)
    ax.set_zlabel('PCA3: Project Quality', labelpad=10)
    ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.view_init(elev=30, azim=-60)
    ax.set_title(f'3D Visualization: {plot_title}')
    plt.show()

    # 2D Scatter Plots
    fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
    axarr[0].scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_data, cmap='tab10', s=50)
    axarr[0].set_xlabel('PCA1: Context and Operations')
    axarr[0].set_ylabel('PCA2: Governance and Management')
    axarr[0].set_title(f'2D Visualization: {plot_title}')

    axarr[1].scatter(pca_data[:, 1], pca_data[:, 2], c=cluster_data, cmap='tab10', s=50)
    axarr[1].set_xlabel('PCA2: Governance and Management')
    axarr[1].set_ylabel('PCA3: Project Quality')

    axarr[2].scatter(pca_data[:, 2], pca_data[:, 0], c=cluster_data, cmap='tab10', s=50)
    axarr[2].set_xlabel('PCA3: Project Quality')
    axarr[2].set_ylabel('PCA1: Context and Operations')

    # Adjust spacing for better visibility
    fig.subplots_adjust(bottom=0.2, hspace=0.4)

    # Display colorbar
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])  # positioning the colorbar below the plots
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal', ticks=np.linspace(0, n_clusters - 1, n_clusters))
    cbar.set_label('Cluster')

    # Adjust tick labels for better centering
    cbar.set_ticks(np.linspace(0, n_clusters - 1, n_clusters))
    cbar.ax.set_xticklabels(range(n_clusters), ha='center')
    
    plt.show()

# Plot all factors
plot_data(tfidf_pca, clusters, 'All Factors')

# Plot only "CSF" and "CFF" factors
plot_data(tfidf_pca[valid_indices], clusters[valid_indices], 'CSF and CFF Factors')


# In[13]:


# SECTION: ANALYSIS OF THE SOLE BULKY CLUSTER 2

# KMeans clusterization of the sole cluster 2

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming clusters is a previously defined array of cluster labels
cluster2_indices = [i for i, c in enumerate(clusters) if c == 2]

# Factors of cluster 2
cluster2_factors = [l_tfidf[i] for i in cluster2_indices]

# Compute TF-IDF matrix for cluster 2
vectorizer_cluster2 = TfidfVectorizer()
tfidf_matrix_cluster2 = vectorizer_cluster2.fit_transform(cluster2_factors)

# Perform KMeans on cluster 2
num_clusters = 10  # Specifying to create 10 clusters
kmeans_cluster2 = KMeans(n_clusters=num_clusters, random_state=42)  # Adjust the number of subclusters as needed
subclusters = kmeans_cluster2.fit_predict(tfidf_matrix_cluster2)

for i in range(max(subclusters) + 1):
    print(f"Subcluster {i}: {sum(subc == i for subc in subclusters)} factors")


# In[14]:


# SECTION: ANALYSIS OF THE SOLE BULKY CLUSTER 2



# Cluster 2: export of factors by subcluster

df_cluster2_factors = pd.DataFrame({
    'Factor': cluster2_factors,
    'Subcluster': subclusters
})

# Save to Excel
df_cluster2_factors.to_excel("/home/nicola/Documenti/PhD_paper_1/cluster2-factors.xlsx", index=False)



# Cluster 2: Identify and Export Top 20 Bigrams

vectorizer_bigrams = CountVectorizer(ngram_range=(2, 2), max_features=20)
bigrams = vectorizer_bigrams.fit_transform(cluster2_factors)
bigrams_df = pd.DataFrame(bigrams.toarray(), columns=vectorizer_bigrams.get_feature_names_out())

# Save to Excel
bigrams_df.sum().sort_values(ascending=False).to_excel("/home/nicola/Documenti/PhD_paper_1/cluster2-top-bigrams.xlsx")



# Cluster 2: Identify and Export Top 5 Words by Subcluster

top_words_by_subcluster = []

for i in range(max(subclusters) + 1):
    subcluster_factors = [factor for j, factor in enumerate(cluster2_factors) if subclusters[j] == i]
    vectorizer_subcluster = TfidfVectorizer(max_features=5)
    tfidf_subcluster = vectorizer_subcluster.fit_transform(subcluster_factors)
    top_words = vectorizer_subcluster.get_feature_names_out()
    top_words_by_subcluster.append(top_words)

df_top_words = pd.DataFrame(top_words_by_subcluster).T

# Save to Excel
df_top_words.to_excel("/home/nicola/Documenti/PhD_paper_1/cluster2-subcluster-topwords.xlsx", index=False, header=[f"Subcluster {i}" for i in range(df_top_words.shape[1])])


# In[15]:


# ANCILLARY: ZOOM IN IN THE 3D SCATTER PLOT

fig_zoomed = plt.figure(figsize=(10, 8)) # Set figure size to ensure all labels fit
ax_zoomed = fig_zoomed.add_subplot(111, projection='3d')

# Plot the data with colors based on clusters, using the "tab10" colormap for distinct colors
scatter_zoomed = ax_zoomed.scatter(tfidf_pca[:, 0], tfidf_pca[:, 1], tfidf_pca[:, 2], c=clusters, cmap='tab10')

# Add axis labels
ax_zoomed.set_xlabel('PC1', labelpad=10)
ax_zoomed.set_ylabel('PC2', labelpad=10)
ax_zoomed.set_zlabel('PC3', labelpad=10)

# Set the zoomed-in limits for each axis
ax_zoomed.set_xlim(-0.2, 0.1)
ax_zoomed.set_ylim(0.0, 0.3)
ax_zoomed.set_zlim(-0.2, 0.1)

# Add a colorbar to represent the clusters
plt.colorbar(scatter_zoomed, ax=ax_zoomed, label='Cluster')

# Show the zoomed-in plot
plt.show()


# In[16]:


# SECTION 5, CLUSTERING BY AUTHOR AND CRITICALITY

# PART 1


print("\nCLUSTERING BY AUTHOR AND CRITICALITY\n")

# Define the references to be consider aed
references_to_consider = [
    "hollow_2011", "ADB_IED_2016", "AfDB_2019", "IFAD_2018", "WB_2011", "IMF_IEO_2017", "mcnamara_2020"
]

# Filter the DataFrame by the class (CSF or CFF) and the required references
df_filtered = df_literature[(df_literature['class'].isin(['CSF', 'CFF'])) &
                            (df_literature['reference'].isin(references_to_consider))]

# Prepare the text by tokenizing, lowercasing, removing punctuation, and lemmatizing
lemmatizer = WordNetLemmatizer()
stop_words_set = set(stopwords.words('english'))

# Create a dictionary to store the top 5 words for each reference
top_words_by_reference = defaultdict(list)

# Create a TfidfVectorizer with custom parameters if needed
# vectorizer = TfidfVectorizer(stop_words='english')  # Modify parameters as needed

# Create a TfidfVectorizer with custom parameters to capture Ngrams
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 2))  # Bi or Trigrams only

# For each reference, calculate the TF-IDF scores
for reference in references_to_consider:
    # Filter the DataFrame to only include factors from the current reference
    factors_reference = df_filtered[df_filtered['reference'] == reference]['factor']
    
    # Preprocess factors
    processed_factors = []
    for factor in factors_reference:
        tokens = word_tokenize(factor)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words_set and word not in string.punctuation]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_factors.append(" ".join(tokens))

    # Remove the three most frequent words
    all_words = " ".join(processed_factors).split()
    most_common_words = [word for word, _ in Counter(all_words).most_common(3)]
    processed_factors = [" ".join([word for word in factor.split() if word not in most_common_words]) for factor in processed_factors]

    # Calculate the TF-IDF matrix for factors in this reference
    tfidf_matrix = vectorizer.fit_transform(processed_factors)

    # Get the feature names (words) in the TF-IDF matrix
    feature_names = vectorizer.get_feature_names_out()
    # Sum the TF-IDF score for each word
    word_scores = tfidf_matrix.sum(axis=0).A1
    # Create a dictionary to store word-score pairs
    word_score_dict = dict(zip(feature_names, word_scores))
    # Sort the words by reverse TF-IDF score
    sorted_words = sorted(word_score_dict.items(), key=lambda x: x[1], reverse=True)
    # Get the top 5 words with the highest TF-IDF scores
    top_words = sorted_words[:10]
    # Store the top 10 words for this reference
    top_words_by_reference[reference] = top_words

# Print the top 5 words for each reference
for reference, top_words in top_words_by_reference.items():
    print(f'Reference {reference}: {top_words}\n')

# Convert the top words by reference into a DataFrame
top_words_df = pd.DataFrame([(reference, word, score) for reference, top_words in top_words_by_reference.items() for word, score in top_words], columns=['Reference', 'Word', 'TF-IDF Score'])

# Save the DataFrame to an Excel file
output_file = 'top_words_by_reference.xlsx'
top_words_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")


# In[ ]:


# GRAPHICAL ANALYSIS OF BIGRAMS - I AM NOT ABLE TO OVERCOME THE ERROR ValueError: Only supported for TrueType fonts


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Example bigrams
bigrams_dict = {'critical success': 50, 'success factor': 30, 'factor analysis': 20}

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigrams_dict)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

