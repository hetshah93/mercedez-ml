from flask import Flask
from flask import render_template
from flask import request
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import xgboost as xgb

app = Flask('__name__', template_folder='templates')
xgb_model=pickle.load(open('xgb_model.pkl','rb'))
stacked_model=pickle.load(open('stacked_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      df = pd.read_csv(f)
      df = pd.get_dummies(df)
      print(df.shape)
      X_test = np.array(df)      
      print(X_test.shape)
      final_predictions = 0.725 * xgb_model.predict(xgb.DMatrix(X_test)) + 0.275 *stacked_model.predict(X_test)
      print('uploaded the file')
      print(final_predictions)
      return render_template('index.html',prediction_text='Time in seconds will be Rs. {}'.format(int(final_predictions)))

if(__name__=='__main__'):
    app.run(debug=True)

