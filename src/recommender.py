import pandas as pd
import numpy as np
from src.recommender import make_tfidf_matrix, get_recommendations, get_indices, get_cos_sim_recs, show_products
from src.latent_dirichlet import print_top_words, run_lda, get_lda_recs
from src.cluster import cluster_text, get_kmeans_rec, plot_elbow, plot_dendro_and_clusters, show_products
from src.split_prices import split_prices

pd.set_option('display.max_columns', 500)
products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)

index_of_item = input('Please enter the index of your item: ')

print('Your chosen item is {}'.format(products['product_title'].iloc[index_of_item]))

# 
# dfs = split_prices(products)
# row_indices, index_of_item,index_df = get_indices(dfs[3],20000)
# cos_item_indices = get_cos_sim_recs(dfs[3], row_indices, index_of_item, index_df, num=3)
# lda_item_indices = get_lda_recs(dfs[3],'combo', row_indices, index_of_item,index_df, num=3)
# vectorizer, tfidf_model, kmeans = cluster_text(dfs[3], row_indices)
# recs = get_kmeans_rec(dfs[3],row_indices, index_of_item,  kmeans, num=3)
# show_products(dfs[3],index_of_item,lda_item_indices)
