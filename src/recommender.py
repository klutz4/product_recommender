import pandas as pd
import numpy as np
from src.nlp_rec import make_tfidf_matrix, get_indices, get_cos_sim_recs, show_products
from src.latent_dirichlet import get_lda_recs
from src.kmeans_rec import cluster_text, get_kmeans_rec
from src.split_price import split_prices

pd.set_option('display.max_columns', 500)
products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)
dfs = split_prices(products)

index_of_our_item = int(input('Please enter the index of your item (up to 292225): '))
item = products['product_title'].iloc[index_of_our_item]
item_id = products['vendor_variant_id'].iloc[index_of_our_item]
price = products['sale_price'].iloc[index_of_our_item]
print('Your chosen item is {}, which costs ${}'.format(item,price))
print('\n')

price_range = input('What is your price range?\n (Please enter your range as min-max) :')

nums = price_range.split('-')
min = int(nums[0])
max = int(nums[1])
df = products.copy()
df = products[products['sale_price'] > min]
df = df[df['sale_price'] < max]
df.reset_index(inplace=True,drop=True)
item_index = df[df['vendor_variant_id'] == item_id].index.item()

row_indices, index_of_item, index_df = get_indices(df,20000)
row_indices = np.append(row_indices, item_index)

method = input('Would you like to use NLP, LDA, or Kmeans? ')
if method == 'NLP':
    cos_item_indices = get_cos_sim_recs(df, row_indices, item_index, index_df, num=3)
    show_products(df,item_index,cos_item_indices)
elif method == 'LDA':
    lda_item_indices = get_lda_recs(df,'combo', row_indices, item_index,index_df, num=3)
    show_products(df,item_index,lda_item_indices)
elif method == 'Kmeans':
    vectorizer, tfidf_model, kmeans = cluster_text(df, row_indices)
    recs = get_kmeans_rec(df,row_indices, item_index, kmeans, num=3)
    show_products(dfs[3],item_index,recs)
else:
    print("That's not an option. Goodbye.")
