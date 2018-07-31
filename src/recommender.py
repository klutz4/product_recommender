import pandas as pd
import numpy as np
from src.nlp_rec import make_tfidf_matrix, get_indices, get_cos_sim_recs, show_products
# from latent_dirichlet import print_top_words, run_lda, get_lda_recs
from src.kmeans_rec import cluster_text, get_kmeans_rec
from src.split_price import split_prices

pd.set_option('display.max_columns', 500)
products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)
dfs = split_prices(products)

index_of_our_item = int(input('Please enter the index of your item (up to 292225): '))
item = products['product_title'].iloc[index_of_our_item]
print('Your chosen item is {}'.format(item))
print('\n')

price_range = input('What is your price range? Please choose from one of the options below:\n 0-50 \n 50-150\n 100-500\n 400-900\n 750-1500\n 1000-2500\n 5000-6500\n 6000-7500\n 7000-8500\n 8000-9500\n 9000-10500\n 10000-20000\n ')

#get correct df
# price_range - parse out min and max
df = dfs[3]
row_indices, index_of_item,index_df = get_indices(df,20000)

method = input('Would you like to use NLP, LDA, or Kmeans? ')
if method == 'NLP':
    cos_item_indices = get_cos_sim_recs(df, row_indices, index_of_our_item, index_df, num=3)
    show_products(df,index_of_our_item,cos_item_indices)
elif method == 'LDA':
    lda_item_indices = get_lda_recs(df,'combo', row_indices, index_of_our_item,index_df, num=3)
    show_products(df,index_of_our_item,lda_item_indices)
elif method == 'Kmeans':
    vectorizer, tfidf_model, kmeans = cluster_text(df, row_indices)
    recs = get_kmeans_rec(df,row_indices, index_of_our_item, kmeans, num=3)
    show_products(dfs[3],index_of_our_item,recs)
else:
    print("That's not an option. Goodbye.")
