import pandas as pd

import xgboost
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression

from utils import *

FEATURES = ['proj1', 'proj2', 'proj3', 'proj4', 'proj5', 'proj6', 'proj7']
Y_COL = 'points'
K = 10
ALPHAS = [0.01, 0.1, 1, 10, 100]
TRAIN_FILE = 'csvs/train.csv'
DEV_FILE = 'csvs/dev.csv'
TEST_FILE = 'csvs/test.csv'


def get_model_error(model, x_train, y_train, x_test, y_test):
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  mse = calculate_error(pred, y_test)
  return mse

def calculate_error(pred, actual):
  results = pd.Series(pred.astype(float) - actual.astype(float))
  return results.map(lambda x: x**2).mean()


def calculate_mean(row, prune=0):
  vals = list(row)[:-1]
  vals.sort()
  return round(pd.Series(vals[prune:len(vals)-prune]).mean(), 2)

def calculate_median(row):
  vals = list(row)[:-1]
  vals.sort()
  return round(pd.Series(vals).median(), 2)


def main():
  connection = get_connection()
  train_df = pd.read_csv(TRAIN_FILE)
  dev_df = pd.read_csv(DEV_FILE)
  test_df = pd.read_csv(TEST_FILE)

  # We can combine train and dev since libraries automatically
  # handle parameter tuning
  train_dev = pd.concat([train_df, dev_df])

  X_train = train_dev[FEATURES].values
  y_train = train_dev[Y_COL].values
  X_test = test_df[FEATURES].values
  y_test = test_df[Y_COL].values

  # MSE for projections
  for proj in FEATURES:
    mse = calculate_error(test_df[proj], y_test)
    print '{} error: {}'.format(proj, round(mse, 2))

  # MSE on naive projections
  test_df['mean'] = [calculate_mean(r) for ind, r in test_df.iterrows()]
  test_df['mean_prune1'] = [calculate_mean(r,prune=1) for ind, r in test_df.iterrows()]
  test_df['mean_prune2'] = [calculate_mean(r,prune=2) for ind, r in test_df.iterrows()]
  test_df['median'] = [calculate_median(r) for ind, r in test_df.iterrows()]

  naive_combos = ['mean', 'mean_prune1', 'mean_prune2', 'median']
  for combo in naive_combos:
    mse = calculate_error(test_df[combo], y_test)
    print '{} error: {}'.format(combo, round(mse, 2))

  # Initial Tests
  xgb = xgboost.XGBRegressor()
  svr = SVR()
  lr = LinearRegression()
  models = {'xgb': xgb, 'svr': svr, 'lr': lr}
  for model_name, model in models.iteritems():
    mse = get_model_error(model, X_train, y_train, X_test, y_test)
    print '{} error: {}'.format(model_name, round(mse, 2))

  # Linear Tests
  for intercept in [False, True]:
    lr = LinearRegression(fit_intercept=intercept)
    ridge = RidgeCV(cv=K, alphas=ALPHAS, fit_intercept=intercept)
    lasso = LassoCV(cv=K, alphas=ALPHAS, fit_intercept=intercept)
    models = {'lr': lr, 'ridge': ridge, 'lasso': lasso}
    for model_name, model in models.iteritems():
      mse = get_model_error(model, X_train, y_train, X_test, y_test)
      if not intercept:
        model_name += ' (No Intercept)'
      print '{} error: {}'.format(model_name, round(mse, 2))
