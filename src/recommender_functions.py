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

def get_indices(df,sample_size):
    '''
    Input:
        df = original dataframe
        sample_size = desired sample size
    Output:
        row_indices = indices for subset of the df
        item_index = index of the item for which we want similar items
        index_df = dataframe containing the updated index from the subsetted df and vendor_variant_id.'''
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
    '''Get recommendations using NLP and cosine similarity.
    Input:
        df = original dataframe
        row_indices = indices for subset of the df
        item_index = index of the item for which we want similar items
        index_df = dataframe containing the updated index from the subsetted df and vendor_variant_id
        starting_point = where to start choosing similar scores (default = 1)
        num = number of recommendations

    Output:
        recommendations from the get_recommendations function
    '''
    stop_words=set(stopwords.words('english'))
    for col in df.columns.values:
        stop_words.add(col)
    tfidf = TfidfVectorizer(analyzer = 'word', lowercase=True, stop_words=stop_words)
    tfidf_matrix = tfidf.fit_transform(df['combo'].iloc[row_indices])
    cosine_sim = linear_kernel(tfidf_matrix)
    item = df['vendor_variant_id'].iloc[item_index]
    print('Cosine Similarity:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[item_index] + "...")
    print("-------")
    return get_recommendations(df, item, index_df, cosine_sim, starting_point,num)

#functions for Kmeans clustering
def cluster_text(df,row_indices):
    '''Use MiniBatchKMeans clustering on the data.
    Input:
    df = original dataframe
    row_indices = indices for subset of the df

    Output:
    vectorizer = tfidf vectorizer
    tfidf_model = fit tfidf model
    kmeans = MiniBatchKMeans model
    '''
    stop_words=set(stopwords.words('english'))
    for col in df.columns.values:
        stop_words.add(col)
    data = df['combo'].iloc[row_indices]
    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=WordNetLemmatizer().lemmatize, lowercase=True)
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

def get_kmeans_rec(df, row_indices, item_index, index_df, kmeans, num=5):
    '''
    Input:
        df = original dataframe
        row_indices = indices for subset of the df
        item_index = index of the item for which we want similar items
        index_df = dataframe containing the updated index from the subsetted df and vendor_variant_id
        index_df = dataframe containing the updated index from the subsetted df and vendor_variant_id
        num = number of recommendations

    Output:
        recs = indices of n recommendations from the same cluster
    '''
    item_id = df['vendor_variant_id'].iloc[item_index]
    subset_index = index_df[item_id]
    cluster_label = kmeans.labels_[subset_index]
    cluster_members = df.iloc[row_indices][kmeans.labels_ == cluster_label]
    recs = np.random.choice(cluster_members.index, num, replace = False)
    print('Mini Batch KMeans:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[item_index] + "...")
    print("-------")
    for rec in recs:
        print("Recommended: " + df['product_title'].iloc[rec] + "\nPrice: $" + str(df['sale_price'].iloc[rec]))

    return recs

#functions for LDA clustering
def print_top_words(model, feature_names, n_top_words=10):
    '''
    Prints the top words for each LDA topic.
    '''
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def run_lda(df,row_indices):
    '''
    Perform LDA on a given dataframe. Returns the fit LDA matrix.
    Input:
        df = original dataframe
        row_indices = indices for subset of the df
    Output:
        lda_matrix = fit LDA matrix'''
    stop_words=set(stopwords.words('english'))
    for col in df.columns.values:
            stop_words.add(col)
    tf_vectorizer = CountVectorizer(analyzer = 'word', stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(df['combo'].iloc[row_indices])
    lda = LatentDirichletAllocation(batch_size=100, n_jobs=-1,max_iter=10, learning_method='online', random_state=0)
    lda_matrix = lda.fit_transform(tf)
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names)
    return lda_matrix

def get_lda_recs(df,row_indices, item_index,index_df,starting_point=1, num=5):
    '''Get recommendations based on the LDA clustering.
    Input:
        df = original dataframe
        row_indices = indices for subset of the df
        item_index = index of the item for which we want similar items
        index_df = dataframe containing the updated index from the subsetted df and vendor_variant_id
        starting_point = where to start choosing similar scores (default = 1)
        num = number of recommendations

    Output:
        recommendations from the get_recommendations function
    '''
    matrix = run_lda(df,row_indices)
    cos_sim = cosine_similarity(matrix)
    item = df['vendor_variant_id'].iloc[item_index]
    print('LDA:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[item_index] + "...")
    print("-------")
    return get_recommendations(df, item, index_df, cos_sim,starting_point,num)

#recommendation functions
def get_recommendations(df,item_id, index_df, cosine_sim,starting_point=1,num=5):
    ''' Return the titles and price of top items with the closest cosine similarity.
    Input:
        df = original dataframe
        item_id - vendor_variant_id for desired item
        index_df = dataframe containing the updated index from the subsetted df and vendor_variant_id
        cosine_sim = cosine similarity matrix
        starting_point = where to start choosing similar scores (default = 1)
        num = number of recommendations

    Output:
        item_indices = indices of the closest items
        '''
    idx = index_df[item_id]
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

def show_products(df, item_index, item_indices):
    '''Opens the weblinks for the specified products.
    Input:
        df = original dataframe
        item_index = index of the item for which we want similar items
        item_indices = indices of the recommended products.
    Output:
        None
        '''
    if len(item_indices) == 0:
        print("Looks like your item is unique! Nothing is similar!")
    else:
        for idx in item_indices:
            webbrowser.open(df['weblink'].iloc[idx], new=1)
