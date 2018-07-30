import pandas as pd
from src.recommender import make_tfidf_matrix, get_recommendations
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import numpy as np
np.random.seed(2018)

def run_nmf(indices):
    tfidf_vectorizer, tfidf_matrix = make_tfidf_matrix(products,'product_title',indices)[0],make_tfidf_matrix(products,'product_title',indices)[1]
    nmf = NMF(max_iter = 5).fit(tfidf_matrix)
    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)
    return nmf.fit_transform(tfidf_matrix)

def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def run_lda(df):
    '''Perform LDA on a given dataframe'''
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(df)
    lda = LatentDirichletAllocation(batch_size=1000, n_jobs=-1, max_iter=5, learning_method='online', random_state=0)
    lda_matrix = lda.fit_transform(tf)
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names)
    return lda_matrix

def get_recs(df,col):
    '''Get recommendations based on the LDA clustering or NMF clustering'''
    matrix = run_lda(df[col])
    row_indices = np.random.choice(292225, 20000)
    cos_sim = cosine_similarity(matrix[row_indices,:],matrix[row_indices,:])
    index_of_item_id = np.random.choice(20000)
    df_index = pd.Series(products.index, index=products['vendor_variant_id']).drop_duplicates()
    return get_recommendations(df,df['vendor_variant_id'].iloc[index_of_item_id],index_of_item_id,df_index, cos_sim, num=10)

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    get_recs(products,'combo')
    # get_recs(run_lda, products,'combo')
    # get_recs(run_nmf,products,'combo')
