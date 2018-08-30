from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np


@app.route('/', methods =['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/nlp_recs', methods=['GET','POST'])
def nlp():
    pass

@app.route('/neural_net_recs', methods=['GET','POST'])
def neural_net_recs():
    pass

if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
