import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


def cluster_text(data):
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True)
    tfidf_model = vectorizer.fit_transform(data)

    kmeans = MiniBatchKMeans(n_clusters=20).fit(tfidf_model)
    y_kmeans = kmeans.predict(tfidf_model)
    centroids = kmeans.cluster_centers_

    for cluster in centroids:
        sorted_cluster = sorted(cluster,reverse=True)
        top_ten = sorted_cluster[:10]
        indices = np.argsort(cluster)[::-1][:10]
        names = []
        for idx in indices:
            names.append(vectorizer.get_feature_names()[idx])
        print(names)
    return vectorizer, tfidf_model, kmeans

def get_kmeans_rec(kmeans, item_id_idx):
    cluster_label = kmeans.labels_[item_id_idx]
    cluster_members = products[kmeans.labels_ == cluster_label]
    recs = np.random.choice(cluster_members.index, 10, replace = False)
    print("Ten recommendations for " + products['product_title'].iloc[item_id_idx] + ":")
    for rec in recs:
        print("Recommended: " + products['product_title'].iloc[rec] + "\nPrice: $" + str(products['sale_price'].iloc[rec]) + "\nWeb link: " + products['weblink'].iloc[rec])


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    # indices = np.random.choice(292225, 10000)
    idx = np.random.choice(292225,1)
    vectorizer, tfidf_model, kmeans = cluster_text(products['combo'])
    # get_kmeans_rec(kmeans,idx)

    #products from first cluster
    # products['taxonomy_name'].iloc[indices][kmeans.labels_ == 0]

    #hierarchical clustering - only works with subset
    # sim = pdist(tfidf_model.toarray())
    # sim_matrix = squareform(sim)
    # hierarchies = linkage(sim,'complete','cosine')
    # dendro = dendrogram(hierarchies)
    # plt.show()
