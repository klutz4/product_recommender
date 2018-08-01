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
from src.nlp_rec import get_indices, show_products
import autoreload


def cluster_text(df,row_indices):
    data = df['combo'].iloc[row_indices]
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True)
    tfidf_model = vectorizer.fit_transform(data)

    kmeans = MiniBatchKMeans(n_clusters=50).fit(tfidf_model)
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

def get_kmeans_rec(df, row_indices, index_of_item, sub_index_of_item, kmeans, num=5):
    cluster_label = kmeans.labels_[sub_index_of_item]
    cluster_members = df.iloc[row_indices][kmeans.labels_ == cluster_label]
    recs = np.random.choice(cluster_members.index, num, replace = False)
    print('Mini Batch KMeans:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[index_of_item] + "...")
    print("-------")
    for rec in recs:
        print("Recommended: " + df['product_title'].iloc[rec] + "\nPrice: $" + str(df['sale_price'].iloc[rec]))

    return recs

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    row_indices, item_index, index_df = get_indices(products,20000)
    vectorizer, tfidf_model, kmeans = cluster_text(products, row_indices)
    recs = get_kmeans_rec(df, row_indices, item_index, item_index,  kmeans, num=5)
