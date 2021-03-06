# -*- coding: utf-8 -*-
"""Road-Friction-Forecasting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W15eOQbeHp9wbJWRaE0f7ZfYv_jAj14O

# Authors: 
**Md. Abrar Jahin**

*   LinkedIn: https://www.linkedin.com/in/md-abrar-jahin-9a026018b
*   Facebook: https://www.facebook.com/
*   Github: https://github.com/Abrar2652
*   email: abrar.jahin.2652@gmail.com


**Andrii Krutsylo**

*  Website: https://krutsylo.neocities.org
*  email: krutsylo@airmail.cc

# Import Libraries and Packages

After importing libraries and packages, we start off by defining a function `transform_to_supervised` that creates desired **lag** *(24 hours in this case)* and **forecasting** features *(1 hour)* of our independent variables concatening with the dataframe and returns the final dataframe.
"""

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
    # Lag features
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

"""# Data Collection and Import data

The dataset has been collected from the **Smart Road - Winter Road Maintenance Challenge 2021** organized by *UiT The Arctic University of Norway* on Devpost.

Dataset download link: https://uitno.app.box.com/s/bch09z27weq0wpcv8dbbc18sxz6cycjt

After downloading the `smart_road_measurements.csv` file from the competition page, wehad added extra columns collecting data from the external resources authorized the organizers. The links of the external datasets are:

[1] Weather data https://pypi.org/project/wwo-hist/ 

[2] UV Index data https://pyowm.readthedocs.io/en/latest/v3/uv-api-usage-examples.html

After merging these 3 files together based on the same dates, we finalized our main dataset `smart_road_measurements_new_d_weather.csv` on top of which we will build our model after preprocessing.
"""

df = pd.read_csv("/content/smart_road_measurements_new_d_weather.csv", header=0)
df2 = df.copy()

df.head(15)

"""# Exploratory Data Analysis

Our dataset contains 349613 rows and 29 columns
"""

df.shape

df.info()

import numpy as np 
np.random.seed(0)
import seaborn as sns
sns.set_theme()
_ = sns.heatmap(df2.iloc[:,2:11].corr())

_ = sns.heatmap(df2.corr())

"""We want to predict Friction of the road by weather conditions. So,
this is a classification task. Every day the car drives on a new route.
This means that all 11 days we receive data on new road sections. So, the
only link between the road sections is the average weather conditions.


This can be achieved by filtering the rows on **Microsoft Excel** for each date and get the total distance covered (the last row on each date because the column is cumulative in nature)

**Max Distance traveled, Date**

42441, 16/02/2021

92311, 17/02/2021

150216, 18/02/2021

39007, 19/02/2021

71358, 22/02/2021

81999, 23/02/2021

55958, 24/02/2021

77315, 25/02/2021

55647, 26/02/2021

61534, 1/03/2021

12409, 2/03/2021


**Therefore, we can see from the above data that for all 11 days the car was driving at different routes**

* We drop the `Distance` because the condition of the road does not depend on how much the car has traveled before. We use this column to get the speed and slope of the road.

* This means that we are using normalized data + lag (time-series
classification with engineered features instead of time-series
classification with deep learning, because we have shallow data). 
We won't focus on any complicated models, just XGBClassifier to win.

* Now we need to define at what Friction the road is dangerous (label 0),
requires caution (label-1) and safe (label-2).

Ta, Tsurf, friction are **highly correlated** which has been shown in our pandas profiling 
https://krutsylo.neocities.org/SmartRoads/pandas3.html of the smart road dataset.

Yet we'll drop State, Height, Distance, Ta, Tsurf, Water, moon-illumination, uvIndex columns
"""

df = df.drop("Height", axis=1) # contain N/A

df = df.drop("Distance", axis=1)
df = df.drop("State", axis=1)
df = df.drop("Ta", axis=1)
df = df.drop("Tsurf", axis=1)
df = df.drop("Water", axis=1)

df = df.drop("moon_illumination", axis=1)
df = df.drop("uvIndex", axis=1)
df.head()

"""  We have grouped the data by calculating the mean of the rows in each hour based on the individual dates. For instance, if there are 8 rows for each hour, we calculated the mean of 8 rows and thus converted into a single row belonging to the distinct dates.

  We also avoided duplicates to reduce the noise in the data.
"""

df['Time(+01:00)'] = pd.to_datetime(df['Time(+01:00)'], format='%H:%M:%S').dt.hour
df = df.groupby(['Date','Time(+01:00)']).mean()
df = df.drop_duplicates()
df.head()

"""Now we will work on the target feature that is `Friction` column to accomplish our objective since we want to perform a supervised machine learning model. Here we applied our knowledge of physics and research capabilities.

Icy: These roads typically have the lowest coefficient of friction. For drivers, this is the most dangerous surface to be on. The small coefficient of friction gives the driver the least amount of traction when accelerating, braking, or turning (which has angular acceleration). Icy roads have a frictional coefficient of around 0.1.

Wet: Roads wet with water have a coefficient of friction of around .4.  This is around 4 times higher than an icy road. Although these roads are much safer to drive on, there is still the possibility of hydroplaning. Hydroplaning occurs when there is standing or flowing water on the road (typically from rainfall) that causes a tire to lose contact with the road's surface. The treads are designed to allow water to fill the crevices so that contact may be maintained between the road and the tire. However, if there is too much water, this may not be achieved, and hydroplaning will occur. This is precisely the reason that racing slicks have such a high coefficient of friction on dry roads (about .9) and a much lower coefficient on wet roads (as low as .1).  

Dry: Roads without precipitation are considered optimal for driving conditions. They have the highest coefficient of friction, around 0.9, which creates the most traction. This allows corners, acceleration, and braking to reach higher values without loss of control. Oftentimes, if roads are not dry, races will be canceled due to the extreme dangers that a less than optimal frictional surface can pose.

So, we'll take (0 <= friction < 0.5) as *dangerous*, and (0.5 < friction <= 1) as *safe*
"""

bins = [0, 0.5, 1]
labels = [0, 1]
df["Friction"] = pd.cut(df["Friction"], bins, labels=labels)

#df = df.drop("Date", axis=1)
#df = df.drop("Time(+01:00)", axis=1)
df.head()

"""Now we'll perform lagging and forecasting feature columns by shifting simply using our pre-defined `transform_to_supervise` function."""

df = transform_to_supervised(df, previous_steps=24, forecast_steps=1, dropnan=True)

Y = df.loc[:, "Friction(t)"].to_numpy()

cols = [c for c in df.columns if '(t)' not in c]
data=df[cols]

data['Friction'] = Y
data.to_csv('/content/test.csv')
data = data.values.tolist()

df[cols].head()

"""**Lag of 1 to 3 days**"""

lag = pd.read_csv('/content/lag(1-3)days.csv')
lag=lag.head(10)
lag

ax = lag.plot(x="Date", y="humidity(t-3)", kind="bar")
lag.plot(x="Date", y="humidity(t-2)", kind="bar", ax=ax, color="C2")
lag.plot(x="Date", y="humidity(t-1)", kind="bar", ax=ax, color="C3")

ax = lag.plot(x="Date", y="windspeedKmph(t-3)", kind="bar")
lag.plot(x="Date", y="windspeedKmph(t-2)", kind="bar", ax=ax, color="C2")
lag.plot(x="Date", y="windspeedKmph(t-1)", kind="bar", ax=ax, color="C3")

"""# Statistical Analysis

**Mean values of each column**
"""

mean = pd.read_csv('/content/Mean.csv')
mean.head()

ax = mean.plot(x="Date", y="windspeedKmph", kind="bar")
mean.plot(x="Date", y="DewPointC", kind="bar", ax=ax, color="C2")
mean.plot(x="Date", y="tempC", kind="bar", ax=ax, color="C3")
#mean.plot(x="Date", y="HeatIndexC", kind="bar", ax=ax, color="C4")
#mean.plot(x="Date", y="humidity", kind="bar", ax=ax, color="C4")
#mean.plot(x="Date", y="pressure", kind="bar", ax=ax, color="C5")

plt.show()

mean.columns

mean.plot(x="Date", y=['tempC'], kind="bar")

mean.plot(x="Date", y=['windspeedKmph'], kind="bar")

mean.plot(x="Date", y=['humidity'], kind="bar")



"""**Standard Deviations of each column**"""

stdev=pd.read_csv('/content/Stdev.csv')
stdev

stdev.plot(x="Date", y=['humidity'], kind="bar")

"""**Minimum values of each column**"""

pd.read_csv('/content/Min.csv')

"""**Maximum values of each column**"""

pd.read_csv('/content/Max.csv')

"""**Median values of each column**"""

med = pd.read_csv('/content/Median.csv')
med

ax = med.plot(x="Date", y="windspeedKmph", kind="bar")
med.plot(x="Date", y="DewPointC", kind="bar", ax=ax, color="C2")
med.plot(x="Date", y="tempC", kind="bar", ax=ax, color="C3")
#mean.plot(x="Date", y="HeatIndexC", kind="bar", ax=ax, color="C4")
#med.plot(x="Date", y="humidity", kind="bar", ax=ax, color="C4")
#med.plot(x="Date", y="pressure", kind="bar", ax=ax, color="C5")

"""**Quartile-1 values of each column**

25% data are less than these values
"""

pd.read_csv('/content/Q1.csv')

"""**Quartile-3 values of each column**

75% data are less than these values
"""

pd.read_csv('/content/Q3.csv')

"""**Kurtosis values of each column**

"""

krt = pd.read_csv('/content/Kurtosis.csv')
krt

ax = krt.plot(x="Date", y="windspeedKmph", kind="bar")
krt.plot(x="Date", y="DewPointC", kind="bar", ax=ax, color="C2")
krt.plot(x="Date", y="tempC", kind="bar", ax=ax, color="C3")
krt.plot(x="Date", y="humidity", kind="bar", ax=ax, color="C4")
krt.plot(x="Date", y="pressure", kind="bar", ax=ax, color="C5")

plt.show()

ax = med.plot(x="Date", y="windspeedKmph", kind="bar")
med.plot(x="Date", y="DewPointC", kind="bar", ax=ax, color="C2")
med.plot(x="Date", y="tempC", kind="bar", ax=ax, color="C3")

"""**Skewness values of each column**

"""

skw=pd.read_csv('/content/Skewness.csv')
skw

ax = skw.plot(x="Date", y="windspeedKmph", kind="bar")
skw.plot(x="Date", y="DewPointC", kind="bar", ax=ax, color="C2")
skw.plot(x="Date", y="tempC", kind="bar", ax=ax, color="C3")
skw.plot(x="Date", y="humidity", kind="bar", ax=ax, color="C4")
skw.plot(x="Date", y="pressure", kind="bar", ax=ax, color="C5")

plt.show()

"""# Model Building, Training and Prediction"""

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
df.to_csv("/content/res.csv", sep="\t")
print(study.best_trial)

"""# Model Evaluation

Our model's Weighted accuracy is 0.7666666666666666 or **~76.67%**.
Now our model can forecast the friction and will label it as **'0' if dangerous** and **'1' if safe** for driving.
"""