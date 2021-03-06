import pandas as pd
import numpy as np
from src.recommender_functions import get_indices, show_products,cluster_text, get_kmeans_rec, get_lda_recs, get_cos_sim_recs
import autoreload

pd.set_option('display.max_columns', 500)
products = pd.read_csv('s3://capstone-3/data/products_wo_na.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)
products = products[products['category'] == 'art']

def combine_columns(x):
    '''
    Combine certain columns with strings into one combined string for NLP.
    '''
    return ''.join(x['product_title']) + ' ' + ''.join(x['product_description']) + ' ' + ''.join(x['material'])

products['combo'] = products.apply(combine_columns,axis=1)

price_range = input('What is your price range?\n (Please enter your range as min-max): ')

nums = price_range.split('-')
min = int(nums[0])
max = int(nums[1])
#restrict products to a price range
df = products.copy()
df = products[products['sale_price'] > min]
df = df[df['sale_price'] < max]
df.reset_index(inplace=True,drop=True)

row_indices, item_index, index_df = get_indices(df,len(df))
item = df['product_title'].iloc[item_index]
item_id = df['vendor_variant_id'].iloc[item_index]
#
cos_item_indices = get_cos_sim_recs(df, row_indices, item_index, index_df, num = 1)
show_products(df,item_index,cos_item_indices)

lda_item_indices = get_lda_recs(df, row_indices, item_index ,index_df, num = 1)
show_products(df,item_index,lda_item_indices)

vectorizer, tfidf_model, kmeans = cluster_text(df, row_indices)
# recs = get_kmeans_rec(df,row_indices, item_index, index_df, kmeans, num=1)
show_products(df,item_index,recs)
