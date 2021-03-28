import os
import optuna
import pickle
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score
import numpy as np


def transform_to_supervised(df,
                            previous_steps=1, 
                            forecast_steps=1,
                            dropnan=False):

    """
    https://gist.github.com/monocongo/6e0df19c9dd845f3f465a9a6ccfcef37
    
    
    Transforms a DataFrame containing time series data into a DataFrame
    containing data suitable for use as a supervised learning problem.
    
    Derived from code originally found at 
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    
    :param df: pandas DataFrame object containing columns of time series values
    :param previous_steps: the number of previous steps that will be included in the
                           output DataFrame corresponding to each input column
    :param forecast_steps: the number of forecast steps that will be included in the
                           output DataFrame corresponding to each input column
    :return Pandas DataFrame containing original columns, renamed <orig_name>(t), as well as
            columns for previous steps, <orig_name>(t-1) ... <orig_name>(t-n) and columns 
            for forecast steps, <orig_name>(t+1) ... <orig_name>(t+n)
    """
    
    # original column names
    col_names = df.columns
    
    # list of columns and corresponding names we'll build from 
    # the originals found in the input DataFrame
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(previous_steps, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (col_name, i)) for col_name in col_names]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, forecast_steps):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % col_name) for col_name in col_names]
        else:
            names += [('%s(t+%d)' % (col_name, i)) for col_name in col_names]

    # put all the columns together into a single aggregated DataFrame
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


df = pd.read_csv("smart_road_measurements_new_d_weather.csv", header=0)
df = df.drop("Height", axis=1) # contain N/A

df = df.drop("Distance", axis=1)
df = df.drop("State", axis=1)
df = df.drop("Ta", axis=1)
df = df.drop("Tsurf", axis=1)
df = df.drop("Water", axis=1)

df = df.drop("moon_illumination", axis=1)
df = df.drop("uvIndex", axis=1)


df['Time(+01:00)'] = pd.to_datetime(df['Time(+01:00)'], format='%H:%M:%S').dt.hour
df = df.groupby(['Date','Time(+01:00)']).mean()
df = df.drop_duplicates()

bins = [0, 0.5, 1]
labels = [0, 1]
df["Friction"] = pd.cut(df["Friction"], bins, labels=labels)

#df = df.drop("Date", axis=1)
#df = df.drop("Time(+01:00)", axis=1)

df = transform_to_supervised(df, previous_steps=24, forecast_steps=1, dropnan=True)

Y = df.loc[:, "Friction(t)"].to_numpy()

cols = [c for c in df.columns if '(t)' not in c]
data=df[cols]

data['Friction'] = Y
data.to_csv('test.csv')
data = data.values.tolist()






# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    print(n_test)
    data = np.array(data)
    return data[:-n_test, :], data[-n_test:, :]
    
# walk-forward validation for univariate data
def walk_forward_validation(params, data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(params, history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = balanced_accuracy_score(test[:, -1], predictions)
	return error, test[:, -1], predictions

# fit an xgboost model and make a one step prediction
def xgboost_forecast(params, train, testX):
	# transform list into array
	train = np.array(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBClassifier(**params)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(np.array([testX]))
	return yhat[0]
    
def objective(trial: Trial, data) -> float:
    params = {
        "booster": "gbtree",
        #"tree_method": "gpu_hist",
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.5),
        "colsample_bytree": trial.suggest_discrete_uniform(
            "colsample_bytree", 0.1, 1, 0.01
        ),
        "nthread": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    
    mae, y, yhat = walk_forward_validation(params, data, 20)
    
    return mae


if not os.path.exists('output'):
    os.makedirs('output')
        
study = optuna.create_study(
    direction="maximize", 
    sampler=TPESampler(seed=1337),
    study_name="res",
    storage="sqlite:///output/res.db",
    load_if_exists=True
    )
study.optimize(
    lambda trial: objective(trial, data), 
    n_trials=50, 
    show_progress_bar=True
    )
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.to_csv("output/res.csv", sep="\t")
print(study.best_trial)
