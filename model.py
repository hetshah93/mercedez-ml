import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor

from tpot.builtins import StackingEstimator

import xgboost as xgb


train_df = pd.read_csv("C:\\Users\\003XRJ744\\Desktop\\AI-ML\\input\\train.csv")
test_df = pd.read_csv("C:\\Users\\003XRJ744\\Desktop\\AI-ML\\input\\test.csv")

def process_data (train_dataf, test_dataf, n_components=180):
    train_dataf = pd.get_dummies(train_dataf)
    test_dataf = pd.get_dummies(test_dataf)
    
    train_dataf.drop(np.argmax(train_dataf['y']), axis=0, inplace=True)
    
    y = np.array(train_dataf['y'])
    
    train_dataf.drop(['ID'], axis=1, inplace=True)
    test_dataf.drop(['ID'], axis=1, inplace=True)
    
    train_dataf, test_dataf = train_dataf.align(test_dataf, join='inner', axis=1)
    
    X_train = np.array(train_dataf)
    X_test = np.array(test_dataf)
    
    pca = PCA(n_components=180)
    
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    
    return X_train, X_test, y

X_train, X_test, y = process_data(train_df, test_df, n_components=180)
xgb_params = {
    'eta' : 0.0025,
    'max_depth' : 5,
    'subsample' : 0.85,
    'objective' : 'reg:squarederror',
    'eval_metric' : 'rmse',
    'base_score' : np.mean(y)
}
test_dframe = pd.get_dummies(test_df)
train_dframe.align(test_dframe, join='inner', axis=1)

dtrain = xgb.DMatrix(X_train, y)
dtest = xgb.DMatrix(X_test)


n_boosting_rounds = 1500
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=n_boosting_rounds)


stacked_model = make_pipeline(StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.75, tol=1e-05)),ExtraTreesRegressor(bootstrap=False, max_features=0.6, min_samples_split=20, n_estimators=500))
stacked_model.fit(X_train,y)

final_predictions = 0.725 * xgb_model.predict(dtest) + 0.275 *stacked_model.predict(X_test)                      


predictions = pd.DataFrame()
predictions['ID'] = test_df['ID']
predictions['y'] = final_predictions

predictions.to_csv("final_prediction.csv")