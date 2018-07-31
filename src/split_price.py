import pandas as pd
import numpy as np

def split_prices(df):
    '''Split products into different dataframes for different prices.'''
    price_dict = {0:50, 50:150,100:500,400:900,750:1500,1000:2500,2000:3500,3000:4500,4000:5500,5000:6500,6000:7500,7000:8500,8000:9500,9000:10500,10000:20000}

    dfs = []
    for k,v in price_dict.items():
        prods = df.copy()
        prods = prods[prods['sale_price'] > k]
        locals()['prods_{}_{}'.format(k,v)] = prods[prods['sale_price'] < v]
        dfs.append(locals()['prods_{}_{}'.format(k,v)])
    for df in dfs:
        df.reset_index(inplace=True,drop=True)
    return dfs

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('/Users/Kelly/galvanize/capstones/mod2/data/products_combo.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    dfs = split_prices(products)
