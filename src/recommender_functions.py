import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from nltk.corpus import stopwords
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
import autoreload
import webbrowser

def make_tfidf_matrix(df,col,indices):
    '''Returns a TfIdf matrix using the subsetted data entered.'''
    tfidf = TfidfVectorizer(analyzer = 'word', lowercase=True, stop_words=stopwords.words('english'))
    tfidf_matrix = tfidf.fit_transform(df[col].iloc[indices])
    cosine_sim = linear_kernel(tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

def get_indices(df,sample_size):
    '''Get the max of a random sample of 20,000 rows of the dataframe or all rows and one random index for an item.'''
    if len(df) > sample_size:
        row_indices = np.random.choice(len(df), sample_size,replace=False)
        item_index = np.random.choice(sample_size)
        index_df = pd.Series(np.arange(len(df)), index=df['vendor_variant_id']).drop_duplicates()
    else:
        row_indices= np.arange(len(df))
        item_index = np.random.choice(len(df))
        index_df = pd.Series(df.index, index=df['vendor_variant_id']).drop_duplicates()
    return row_indices, item_index, index_df

#functions for cosine similarity
def get_cos_sim_recs(df,row_indices,item_index,index_df,starting_point=1,num=5):
    '''Get recommendations using NLP and cosine similarity (no clustering).'''
    tfidf_model, tfidf_matrix, cosine_sim = make_tfidf_matrix(df,'combo', row_indices)
    item = df['vendor_variant_id'].iloc[item_index]
    print('Cosine Similarity:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[item_index] + "...")
    print("-------")
    return get_recommendations(df, item, index_df, cosine_sim, starting_point,num)

#functions for Kmeans clustering
def cluster_text(df,row_indices):
    data = df['combo'].iloc[row_indices]
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), tokenizer=WordNetLemmatizer().lemmatize, lowercase=True)
    tfidf_model = vectorizer.fit_transform(data)

    kmeans = MiniBatchKMeans(n_clusters=50, batch_size = 20).fit(tfidf_model)
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

#functions for LDA clustering
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def run_lda(df):
    '''Perform LDA on a given dataframe'''
    tf_vectorizer = CountVectorizer(analyzer = 'word', stop_words=stopwords.words('english'))
    tf = tf_vectorizer.fit_transform(df)
    lda = LatentDirichletAllocation(batch_size=100, n_jobs=-1,max_iter=10, learning_method='online', random_state=0)
    lda_matrix = lda.fit_transform(tf)
    # print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    # print_top_words(lda, tf_feature_names)
    return lda_matrix

def get_lda_recs(df,col,row_indices, item_index,index_df,starting_point=1, num=5):
    '''Get recommendations based on the LDA clustering'''
    matrix = run_lda(df[col].iloc[row_indices])
    cos_sim = cosine_similarity(matrix)
    item = df['vendor_variant_id'].iloc[item_index]
    print('LDA:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[item_index] + "...")
    print("-------")
    return get_recommendations(df, item, index_df, cos_sim,starting_point,num)

#recommendation functions
def get_recommendations(df,item, index_df, cosine_sim,starting_point=1,num=5):
    ''' Return the titles and price of top items with the closest cosine similarity.'''
    idx = index_df[item]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the items based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the n most similar items
    if starting_point > len(sim_scores):
        print("Looks like your item is unique! Nothing is similar!")
    elif num+starting_point <= len(sim_scores):
        sim_scores = sim_scores[starting_point:num+starting_point]
    else:
        sim_scores = sim_scores[starting_point:len(sim_scores)]

    item_indices = [i[0] for i in sim_scores]
    if len(item_indices) == 0:
        print("Looks like your item is unique! Nothing is similar!")
    else:
        for i in range(min(len(item_indices),num)):
            print("Recommended: " + df['product_title'].iloc[item_indices[i]] + "\nPrice: $" + str(df['sale_price'].iloc[item_indices[i]]) + "\n(Cosine similarity: {:.4f})".format(sim_scores[i][1]))
            print('\n')
    return item_indices

def show_products(df, index_of_item, item_indices):
    # webbrowser.open(df['weblink'].iloc[index_of_item], new=1)
    if len(item_indices) == 0:
        print("Looks like your item is unique! Nothing is similar!")
    else:
        for idx in item_indices:
            webbrowser.open(df['weblink'].iloc[idx], new=1)
