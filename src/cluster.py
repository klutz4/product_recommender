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


def cluster_text(data):
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True, max_features=200)
    tfidf_model = vectorizer.fit_transform(data)

    kmeans = MiniBatchKMeans(n_clusters=20).fit(tfidf_model)
    centroids = kmeans.cluster_centers_

    for cluster in centroids:
        sorted_cluster = sorted(cluster,reverse=True)
        top_ten = sorted_cluster[:10]
        indices = np.argsort(cluster)[::-1][:10]
        names = []
        for idx in indices:
            names.append(vectorizer.get_feature_names()[idx])
        # print(names)
    return vectorizer, tfidf_model, kmeans

def get_kmeans_rec(kmeans, item_id_idx):
    cluster_label = kmeans.labels_[item_id_idx]
    cluster_members = products[kmeans.labels_ == cluster_label]
    recs = np.random.choice(cluster_members.index, 10, replace = False)
    print("Ten recommendations for " + products['product_title'].iloc[item_id_idx] + ":")
    for rec in recs:
        print("Recommended: " + products['product_title'].iloc[rec] + "\nPrice: $" + str(products['sale_price'].iloc[rec]) + "\nWeb link: " + products['weblink'].iloc[rec])

def plot_elbow(tfidf_model,filename):
        distortions = []
        K = range(1,200)
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

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    indices = np.random.choice(292225, 20000)
    idx = np.random.choice(292225,1)
    vectorizer, tfidf_model, kmeans = cluster_text(products['combo'].iloc[indices])
    # get_kmeans_rec(kmeans,idx)
    # plot_elbow(tfidf_model,'images/elbow.png')
    # plot_dendro_and_clusters(tfidf_model,'images/dendro.png')
