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
from src.recommender import get_indices


def cluster_text(df,row_indices):
    data = df['combo'].iloc[row_indices]
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

def get_kmeans_rec(df, kmeans, index_of_item, num=5):
    cluster_label = kmeans.labels_[index_of_item]
    cluster_members = df[kmeans.labels_ == cluster_label]
    recs = np.random.choice(cluster_members.index, num, replace = False)
    print('Mini Batch KMeans:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[index_of_item] + "...")
    print("-------")
    for rec in recs:
        print("Recommended: " + df['product_title'].iloc[rec] + "\nPrice: $" + str(df['sale_price'].iloc[rec]))
        # + "\nWeb link: " + df['weblink'].iloc[rec]

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

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    row_indices, index_of_item, index_df = get_indices(products)
    vectorizer, tfidf_model, kmeans = cluster_text(products, row_indices)
    # get_kmeans_rec(products, kmeans,index_of_item,num=10)
    # plot_elbow(tfidf_model,'images/elbow.png')
    # plot_dendro_and_clusters(tfidf_model,'images/dendro.png')
