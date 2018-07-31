import pandas as pd
from src.recommender import make_tfidf_matrix, get_recommendations, get_indices, get_cos_sim_recs
from src.latent_dirichlet import print_top_words, run_lda, get_lda_recs
from src.cluster import cluster_text, get_kmeans_rec, plot_elbow, plot_dendro_and_clusters
import numpy as np

pd.set_option('display.max_columns', 500)
products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)

def split_prices():
    '''Split products into different dataframes for different prices.'''
    price_dict = {0:50, 50:150,100:500,400:900,750:1500,1000:2500,2000:3500,3000:4500,4000:5500,5000:6500,6000:7500,7000:8500,8000:9500,9000:10500,10000:20000}

    dfs = []
    for k,v in price_dict.items():
        prods = products.copy()
        prods = prods[prods['sale_price'] > k]
        locals()['prods_{}_{}'.format(k,v)] = prods[prods['sale_price'] < v]
        dfs.append(locals()['prods_{}_{}'.format(k,v)])
    for df in dfs:
        df.reset_index(inplace=True,drop=True)
    return dfs


if __name__ == '__main__':
    dfs = split_prices()
    row_indices, index_of_item,index_df = get_indices(dfs[3],20000)
    get_cos_sim_recs(dfs[3], row_indices, index_of_item, index_df, num=5)
    get_lda_recs(dfs[3],'combo', row_indices, index_of_item,index_df, num=5)
    vectorizer, tfidf_model, kmeans = cluster_text(dfs[3], row_indices)
    get_kmeans_rec(dfs[3],kmeans,index_of_item, num=5)
    plot_elbow(tfidf_model,'images/elbow_df3.png')
