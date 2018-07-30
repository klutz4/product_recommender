import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


def cluster_text(data):
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_df=0.5, min_df=0.1, lowercase=True,max_features=10)
    tfidf_model = vectorizer.fit_transform(data)

    kmeans = KMeans(n_clusters=10).fit(tfidf_model)
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


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    indices = np.random.choice(292225, 20000)
    vectorizer, tfidf_model, kmeans = cluster_text(products['combo'].iloc[indices])
    sim = pdist(tfidf_model.toarray())
    sim_matrix = squareform(sim)
    hierarchies = linkage(sim,'complete')
    dendro = dendrogram(hierarchies)
    plt.show()
