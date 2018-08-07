import pandas as pd
import numpy as np
from src.recommender_functions import get_indices, get_cos_sim_recs, show_products, get_lda_recs, cluster_text, get_kmeans_rec
import webbrowser
import autoreload

pd.set_option('display.max_columns', 500)
products = pd.read_csv('../data/products_art_only.csv')
products.drop('Unnamed: 0',axis=1, inplace=True)

index_of_our_item = int(input('Please enter the index of your item (up to 236706): '))
item = products['product_title'].iloc[index_of_our_item]
item_id = products['vendor_variant_id'].iloc[index_of_our_item]
price = products['sale_price'].iloc[index_of_our_item]
print('Your chosen item is {}, which costs ${}'.format(item,price))
print('\n')
webbrowser.open(products['weblink'].iloc[index_of_our_item], new=1)

price_range = input('What is your price range?\n (Please enter your range as min-max): ')

nums = price_range.split('-')
min = int(nums[0])
max = int(nums[1])
df = products.copy()
df = products[products['sale_price'] >= min]
df = df[df['sale_price'] < max]
if (price < min) or (price > max):
    df = df.append(products.iloc[index_of_our_item],ignore_index=True)

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

method = input('Would you like to use Cosine Sim, LDA, or Kmeans? ')
rec_num = int(input('How many recommendations would you like? '))
print("This'll take a second...")

def run_recommendations(method,starting_point=1):
    if method.lower() == 'cosine sim':
        cos_item_indices = get_cos_sim_recs(df, row_indices, item_index, index_df, starting_point, rec_num)
        show_products(df,item_index,cos_item_indices)
    elif method.lower() == 'lda':
        lda_item_indices = get_lda_recs(df, row_indices, item_index,index_df, starting_point,rec_num)
        show_products(df,item_index,lda_item_indices)
    elif method.lower() == 'kmeans':
        vectorizer, tfidf_model, kmeans = cluster_text(df, row_indices)
        recs = get_kmeans_rec(df,row_indices, item_index, index_df, kmeans, rec_num)
        show_products(df,item_index,recs)
    else:
        print("That's not an option. Goodbye.")

def rerun_recommendations(rating,starting_point):
    if rating.lower() == 'no':
        print("Let's see if I can get some better options...")
        run_recommendations(method,starting_point)
    elif rating.lower() == 'yes':
        print('Enjoy!')
    else:
        print("I'm sorry. I didn't understand that response.")
        rating = input('Did you like those recommendations? ')
        rerun_recommendations(rating, starting_point)

run_recommendations(method)
methods = ['cosine sim', 'lda', 'kmeans']
if method.lower() not in methods:
    pass
else:
    rating = input('Did you like those recommendations? ')
    starting_point = 1 + rec_num
    rerun_recommendations(rating, starting_point)
