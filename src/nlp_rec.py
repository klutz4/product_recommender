from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import webbrowser

def get_recommendations(df,item, index_of_item, index_df, cosine_sim,num=5):
    idx = index_df[item]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the items based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar items
    sim_scores = sim_scores[1:num+1]
    item_indices = [i[0] for i in sim_scores]
    for i in range(num):
        print("Recommended: " + df['product_title'].iloc[item_indices[i]] + "\nPrice: $" + str(df['sale_price'].iloc[item_indices[i]]) + "\n(Cosine similarity: {:.4f})".format(sim_scores[i][1]))
    print('\n')
    return item_indices

def make_tfidf_matrix(df,col,indices):
    tfidf = TfidfVectorizer(analyzer = 'word', lowercase=True, stop_words=stopwords.words('english'))
    tfidf_matrix = tfidf.fit_transform(df[col].iloc[indices])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

def get_indices(df,sample_size):
    '''Get the max of a random sample of 20,000 rows of the dataframe or all rows and one random index for an item.'''
    if len(df) > sample_size:
        row_indices = np.random.choice(len(df), sample_size,replace=False)
        index_of_item = np.random.choice(sample_size)
        index_df = pd.Series(np.arange(len(df)), index=df['vendor_variant_id']).drop_duplicates()
    else:
        row_indices= np.arange(len(df))
        index_of_item = np.random.choice(len(df))
        index_df = pd.Series(df.index, index=df['vendor_variant_id']).drop_duplicates()

    return row_indices, index_of_item, index_df

def get_cos_sim_recs(df,row_indices,index_of_item,index_df,num=5):
    '''Get recommendations using NLP and cosine similarity (no clustering).'''
    tfidf_model, tfidf_matrix, cosine_sim = make_tfidf_matrix(df,'combo', row_indices)
    item = df['vendor_variant_id'].iloc[index_of_item]
    print('NLP and Cosine Similarity:\n')
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[index_of_item] + "...")
    print("-------")
    return get_recommendations(df,item,index_of_item,index_df, cosine_sim, num)

def show_products(df, index_of_item, item_indices):
    webbrowser.open(df['weblink'].iloc[index_of_item], new=1)
    for idx in item_indices:
        webbrowser.open(df['weblink'].iloc[idx], new=1)

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    row_indices, index_of_item,index_df = get_indices(products,20000)
    item_indices = get_cos_sim_recs(products,row_indices,index_of_item,index_df,num=10)
