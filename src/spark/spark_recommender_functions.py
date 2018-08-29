import pyspark as ps
import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.ml.clustering import LDA, KMeans
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.feature import StopWordsRemover
from nltk.corpus import stopwords
from pyspark.ml.evaluation import ClusteringEvaluator

def tfidf_pipeline():
    tokenizer = Tokenizer(inputCol="combo", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=set(stopwords.words('english')))
    hashingTF = HashingTF(inputCol='filtered', outputCol="rawFeatures", numFeatures=100)
    idf = IDF(inputCol='rawFeatures', outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    return pipeline


def get_kmeans_rec(dataset,item_id,num_recs=5):
    kmeans = KMeans(k=50)
    model = kmeans.fit(dataset)
    result = model.transform(dataset)
    labels = result.select('prediction')
    item_cluster_label = result.filter(col('vendor_variant_id') == str(item_id)).select('prediction').collect()[0][0]
    cluster_members = result.filter(col('prediction') == item_cluster_label)
    return cluster_members.select('product_title').show(num_recs)


def get_kmeans_scores(model):
    # Evaluate clustering by computing Within Set Sum of Squared Errors.
    wssse = model.computeCost(dataset)
    print("Within Set Sum of Squared Errors = " + str(wssse))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))


def lda_recs(dataset):
    # Trains a LDA model.
    lda = LDA(k=10, maxIter=10)
    model = lda.fit(dataset)

    ll = model.logLikelihood(dataset)
    lp = model.logPerplexity(dataset)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))

    # Describe topics.
    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

    # Shows the result
    transformed = model.transform(dataset)
    transformed.show(truncate=False)

if __name__ == '__main__':
    spark = (ps.sql.SparkSession.builder
            .master("local[3]")
            .appName("capstone")
            .getOrCreate()
            )
    sc = spark.sparkContext

    #from local
    # df= pd.read_csv('../data/products_art_only.csv')
    # df.drop('Unnamed: 0', axis=1,inplace=True)
    # spark_df = spark.createDataFrame(df)

    #from s3
    df = pd.read_csv('s3a://capstone-3/data/products_art_only.csv')
    df.drop('Unnamed: 0', axis=1,inplace=True)
    spark_df = spark.createDataFrame(df)

    pipeline = tfidf_pipeline()
    features_df = pipeline.fit(spark_df).transform(spark_df)

    item_index = np.random.choice(len(df))
    item_id = df['vendor_variant_id'].iloc[item_index]
    get_kmeans_rec(features_df,5,item_id)
