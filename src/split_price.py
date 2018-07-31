import pandas as pd
from src.recommender import make_tfidf_matrix, get_recommendations
from src.latent_dirichlet import print_top_words, run_lda, get_recs
from src.cluster import cluster_text, get_kmeans_rec, plot_elbow, plot_dendro_and_clusters
import numpy as np

pd.set_option('display.max_columns', 500)
products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)

price_dict = {0:150,100:500,400:900,750:1500,1000:2500,2000:3500,3000:4500,4000:5500,5000:6500,6000:7500,7000:8500,8000:9500,9000:10500,10000:20000}

dfs = []
for k,v in price_dict.items():
    prods = products.copy()
    prods = prods[prods['sale_price'] > k]
    locals()['prods_{}_{}'.format(k,v)] = prods[prods['sale_price'] < v]
    dfs.append(locals()['prods_{}_{}'.format(k,v)])

indices = np.random.choice(len(dfs[0]), 20000)

index = pd.Series(dfs[0].index, index=dfs[0]['vendor_variant_id']).drop_duplicates()
indices1, cosine_sim2 = make_tfidf_matrix(dfs[0],'combo', indices)[2], make_tfidf_matrix(dfs[0],'combo', indices)[3]
idx = np.random.choice(20000)
get_recommendations(dfs[0],dfs[0]['vendor_variant_id'].iloc[idx],idx,index, cosine_sim2)
