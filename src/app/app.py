from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') #for AWS only
import matplotlib.pyplot as plt
# from flask.ext.mobility import Mobility
# from flask.ext.mobility.decorators import mobile_template

app = Flask(__name__)
# Mobility(app)

def get_restricted_df(price,item_index,range):
    nums = range.split('-')
    min = int(nums[0])
    max = int(nums[1])
    restricted = df.copy()
    restricted = restricted[restricted['sale_price'] >= min]
    restricted = restricted[restricted['sale_price'] < max]
    if (price < min) or (price > max):
        restricted = restricted.append(df.iloc[item_index],ignore_index=True)
    return restricted

@app.route('/', methods =['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/home', methods =['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/nlp', methods=['GET','POST'])
def nlp():
    return render_template('nlp.html')

@app.route('/neural_net', methods=['GET','POST'])
def neural_net():
    return render_template('neural_net.html')

@app.route('/nlp_recs', methods=['GET','POST'])
def nlp_recs():
    item_index= int(request.form['index'])
    range = str(request.form['price'])
    # num_recs = int(request.form['recs'])
    item = df['product_title'].iloc[item_index]
    item_id = df['vendor_variant_id'].iloc[item_index]
    price = df['sale_price'].iloc[item_index]
    restricted = get_restricted_df(price,item_index,range)
    cluster_label = restricted['prediction'].iloc[item_index]
    cluster_members = restricted[restricted['prediction'] == cluster_label]
    recs = np.random.choice(cluster_members.index, 5, replace = False)

    return render_template('nlp_recs.html',recs=recs,df=df,item_index=item_index)

@app.route('/cnn_recs', methods=['GET','POST'])
def cnn_recs():
    item_index= int(request.form['index'])
    recs = [3,8,13,45]
    return render_template('cnn_recs.html',item_index=item_index,recs=recs,images=images)

if  __name__ == '__main__':
    df = pd.read_csv('s3a://capstone-3/data/spark_model.csv')
    images = pd.read_csv('s3a://capstone-3/data/art_only_images.csv')
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
