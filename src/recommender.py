from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

def get_recommendations(df,item_id, index_of_item_id, index, cosine_sim,num=5):
    idx = index[item_id]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the items based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar items
    sim_scores = sim_scores[1:11]
    item_indices = [i[0] for i in sim_scores]
    print("Recommending " + str(num) + " products similar to " + df['product_title'].iloc[index_of_item_id] + "...")
    print("-------")
    for i in range(num):
        print("Recommended: " + df['product_title'].iloc[item_indices[i]] + "\nPrice: $" + str(df['sale_price'].iloc[item_indices[i]]) + "\n(score:" + str(sim_scores[i][1]) + ")")

def make_tfidf_matrix(df,col,indices):
    tfidf = TfidfVectorizer(analyzer = 'word', lowercase=True, stop_words=stopwords.words('english'))
    tfidf_matrix = tfidf.fit_transform(df[col].iloc[indices])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, indices, cosine_sim

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_wo_na.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    indices = np.random.choice(292225, 20000)
    indices1, cosine_sim1 = make_tfidf_matrix(products,'product_title', indices)

    # idx = np.random.choice(10000)
    # get_recommendations(products['vendor_variant_id'].iloc[idx],indices, cosine_sim)
    index = pd.Series(df.index, index=df['vendor_variant_id']).drop_duplicates()
    indices2, cosine_sim2 = make_tfidf_matrix(products,'combo', indices)[2], make_tfidf_matrix(products,'combo', indices)[3]
    idx = np.random.choice(10000)
    get_recommendations(products['vendor_variant_id'].iloc[idx],idx,index, cosine_sim2)
