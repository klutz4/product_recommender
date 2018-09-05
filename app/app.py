from flask import Flask, request, render_template
import pandas as pd
import numpy as np
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
    return restricted

@app.route('/', methods =['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/home', methods =['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/nlp', methods=['GET','POST'])
def nlp():
    choices = np.random.choice(df.index,5,replace=False)
    return render_template('nlp.html',df=df,choices=choices)

@app.route('/neural_net', methods=['GET','POST'])
def neural_net():
    choices = np.random.choice(images.index,5,replace=False)
    return render_template('neural_net.html',images=images,choices=choices)

@app.route('/nlp_recs', methods=['GET','POST'])
def nlp_recs():
    try:
        item_index= int(request.form['index'])
        range = str(request.form['price'])
    except:
        item_index= int(request.args.get('index'))
        range = ''
    if range == '':
        cluster_label = df['prediction'].iloc[item_index]
        cluster_members = df[df['prediction'] == cluster_label]
        recs = np.random.choice(cluster_members.index, 5, replace = False)
        return render_template('nlp_recs.html',recs=recs,df=df,item_index=item_index)
    if range != '':
        # num_recs = int(request.form['recs'])
        price = df['sale_price'].iloc[item_index]
        restricted = get_restricted_df(price,item_index,range)
        if len(restricted) == 0:
            return "There are no recommendations in this price range!"
        else:
            cluster_label = df['prediction'].iloc[item_index]
            cluster_members = restricted[restricted['prediction'] == cluster_label]
            try:
                recs = np.random.choice(cluster_members.index, 5, replace = False)
            except:
                recs = np.random.choice(cluster_members.index, len(cluster_members), replace = False)
            return render_template('nlp_recs.html',recs=recs,df=df,item_index=item_index)

@app.route('/cnn_recs', methods=['GET','POST'])
def cnn_recs():
    try:
        item_index= int(request.form['image'])
    except:
        item_index= int(request.args.get('image'))
    cluster_label = images['label'].iloc[item_index]
    cluster_members = images[images['label'] == cluster_label]
    recs = np.random.choice(cluster_members.index, 5, replace = False)
    return render_template('cnn_recs.html',item_index=item_index,recs=recs,images=images)

if  __name__ == '__main__':
    df = pd.read_csv('s3a://capstone-3/data/spark_model.csv')
    images = pd.read_csv('s3a://capstone-3/data/images_and_labels2.csv')

    app.run(host='0.0.0.0',port=8080, debug=True, threaded=True)
