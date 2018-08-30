from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)



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
    price = str(request.form['price'])
    return render_template('nlp_recs.html')

@app.route('/cnn_recs', methods=['GET','POST'])
def cnn_recs():
    item_index= int(request.form['index'])
    return render_template('cnn_recs.html')

if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
