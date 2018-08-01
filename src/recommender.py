import pandas as pd
import numpy as np
from src.nlp_rec import make_tfidf_matrix, get_indices, get_cos_sim_recs, show_products
from src.latent_dirichlet import get_lda_recs
from src.kmeans_rec import cluster_text, get_kmeans_rec
from src.split_price import split_prices

pd.set_option('display.max_columns', 500)
products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)

price_range = input('What is your price range?\n (Please enter your range as min-max): ')

nums = price_range.split('-')
min = int(nums[0])
max = int(nums[1])
#restrict products to a price range
df = products.copy()
df = products[products['sale_price'] > min]
df = df[df['sale_price'] < max]
df.reset_index(inplace=True,drop=True)

row_indices, item_index, index_df = get_indices(df,20000)
item = df['product_title'].iloc[index_of_item]
item_id = df['vendor_variant_id'].iloc[index_of_item]

# item_index = df[df['vendor_variant_id'] == item_id].index.item()
# subset_item_index = index_df[item_id]

cos_item_indices = get_cos_sim_recs(df, row_indices, item_index, index_df, rec_num = 5)
show_products(df,item_index,cos_item_indices)

lda_item_indices = get_lda_recs(df,'combo', row_indices, item_index ,index_df, rec_num = 5)
show_products(df,item_index,lda_item_indices)

vectorizer, tfidf_model, kmeans = cluster_text(df, row_indices)
recs = get_kmeans_rec(df,row_indices, item_index, item_index, kmeans, rec_num=5)
show_products(df,item_index,recs)
