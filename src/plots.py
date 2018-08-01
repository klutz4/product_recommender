import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist

def plot_elbow(tfidf_model,filename):
        distortions = []
        K = range(1,50)
        for k in K:
            kmeans = MiniBatchKMeans(n_clusters=k)
            kmeans.fit(tfidf_model)
            distortions.append(kmeans.inertia_)

        # Plot the elbow
        plt.plot(K, distortions)
        plt.grid(True)
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.savefig(filename)

def plot_dendro_and_clusters(tfidf_model,filename):
        # hierarchical clustering - only works with subset
        plt.figure(figsize=(35, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        sim = pdist(tfidf_model.toarray())
        sim_matrix = squareform(sim)
        hierarchies = linkage(sim,'complete')
        dendro = dendrogram(hierarchies,leaf_rotation=90.,leaf_font_size=8.)
        plt.savefig(filename)
        k=15
        clusters =fcluster(hierarchies,k, depth=10, criterion='maxclust')
        plt.figure(figsize=(10, 8))
        plt.scatter(hierarchies[:,0], hierarchies [:,1], c=clusters[:len(clusters)-1], cmap='prism')  # plot points with cluster dependent colors
        plt.show()

def plot_categories(df,filename):
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    ax.set_title('Proportions of Product Categories')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Prop')
    ax.gcf().subplots_adjust(bottom=0.15)
    df.plot('index','prop', kind='bar'. ax=ax)
    plt.savefig(filename)

def make_prop_df(df):
     prop_df = (df['category']
                .value_counts(normalize=True)
                .rename('prop')
                .reset_index())
    return prop_df
