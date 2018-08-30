import pandas as pd
import numpy as np
import webbrowser
import autoreload
import pyspark as ps
from spark.spark_recommender_functions import tfidf_pipeline, get_kmeans_rec
from pyspark.sql.types import *
from pyspark.sql.functions import col

spark = (ps.sql.SparkSession.builder
        .master("local[3]")
        .appName("capstone")
        .getOrCreate()
        )
sc = spark.sparkContext

pd.set_option('display.max_columns', 500)
df = pd.read_csv('s3a://capstone-3/data/products_art_only.csv')
df.drop('Unnamed: 0', axis=1,inplace=True)
spark_df = spark.createDataFrame(df)

item_index  = int(input('Please enter the index of your item (up to 236706): '))

item_id = df['vendor_variant_id'].iloc[item_index]
item = spark_df.filter(col('vendor_variant_id') == str(item_id)).select('product_title').collect()[0][0]
price = spark_df.filter(col('vendor_variant_id') == str(item_id)).select('sale_price').collect()[0][0]

print('Your chosen item is {}, which costs ${}'.format(item,price))
print('\n')
# webbrowser.open(products['weblink'].iloc[index_of_our_item], new=1)

price_range = input('What is your price range?\n (Please enter your range as min-max): ')

nums = price_range.split('-')
min = float(nums[0])
max = float(nums[1])
restrict_min = spark_df.where(col('sale_price') > min)
restricted = restrict_min.where(col('sale_price') < max)

if (price < min) or (price > max):
    item_row = spark_df.filter(col('vendor_variant_id') == str(item_id))
    restricted = restricted.union(item_row)

num_recs = int(input('How many recommendations would you like? '))
print("This'll take a second...")

pipeline = tfidf_pipeline()
features_df = pipeline.fit(restricted).transform(restricted)
get_kmeans_rec(features_df, item_id, num_recs)
