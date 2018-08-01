import pandas as pd
import numpy as np
from src.nlp_rec import make_tfidf_matrix, get_indices, get_cos_sim_recs, show_products
from src.latent_dirichlet import get_lda_recs
from src.kmeans_rec import cluster_text, get_kmeans_rec
import autoreload

pd.set_option('display.max_columns', 500)
products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_wo_na.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)
products = products[products['category'] == 'art']
products = products[products['price'] != 0]

def combine_columns(x):
    '''
    Combine certain columns with strings into one string for NLP.
    '''
    return ''.join(x['product_title']) + ' ' + ''.join(x['product_description']) + ' ' + ''.join(x['material'])

products['combo'] = products.apply(combine_columns,axis=1)

index_of_our_item = int(input('Please enter the index of your item (up to 236130): '))
item = products['product_title'].iloc[index_of_our_item]
item_id = products['vendor_variant_id'].iloc[index_of_our_item]
price = products['sale_price'].iloc[index_of_our_item]
print('Your chosen item is {}, which costs ${}'.format(item,price))
print('\n')

price_range = input('What is your price range?\n (Please enter your range as min-max): ')

nums = price_range.split('-')
min = int(nums[0])
max = int(nums[1])
df = products.copy()
df = products[products['sale_price'] > min]
df = df[df['sale_price'] < max]

df.reset_index(inplace=True,drop=True)
item_index = df[df['vendor_variant_id'] == item_id].index.item()

if len(df) > 35000:
    row_indices = np.random.choice(len(df), 35000,replace=False)
    extra = row_indices[0]
    row_indices = row_indices[1:]
    if item_index not in row_indices:
        row_indices = np.append(row_indices,item_index)
    else:
        row_indices = np.append(row_indices, extra)
    index_df = pd.Series(np.arange(35000), index=df['vendor_variant_id'].iloc[row_indices]).drop_duplicates()
else:
    row_indices= np.arange(len(df))
    index_of_item = np.random.choice(len(df))
    index_df = pd.Series(df.index, index=df['vendor_variant_id']).drop_duplicates()

subset_item_index = index_df[item_id]

method = input('Would you like to use Cosine Sim, LDA, or Kmeans? ')
rec_num = int(input('How many recommendations would you like? '))
print("This'll take a second...")

if method == 'Cosine Sim':
    cos_item_indices = get_cos_sim_recs(df, row_indices, item_index, index_df, rec_num)
    show_products(df,item_index,cos_item_indices)
elif method == 'LDA':
    lda_item_indices = get_lda_recs(df,'combo', row_indices, item_index,index_df, rec_num)
    show_products(df,item_index,lda_item_indices)
elif method == 'Kmeans':
    vectorizer, tfidf_model, kmeans = cluster_text(df, row_indices)
    recs = get_kmeans_rec(df,row_indices, item_index, subset_item_index, kmeans, rec_num)
    show_products(df,item_index,recs)
else:
    print("That's not an option. Goodbye.")
