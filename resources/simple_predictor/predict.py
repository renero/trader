# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:32:13 2017

@author: sonaam1234, ecejsrq
https://github.com/sonaam1234/DeepLearningInFinance
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from pandas import DataFrame, read_hdf, to_datetime
from sklearn.preprocessing import StandardScaler

np.random.seed(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def timeseries_build(data_frame: DataFrame):
    time_series = np.asarray(data_frame.Gold)
    time_series = np.atleast_2d(time_series)
    if time_series.shape[0] == 1:
        time_series = time_series.T

    return time_series


def read_data(filename='../data/DeepLearning.h5') -> DataFrame:
    """
    The .h5 file contains gold, DJI, InterestRate, Inflation and Reserves
    fields since 1985-02-01 until 2017-02-01, monthly values.
    :return:
    """
    print('Reading data from file: {}'.format(filename))
    return DataFrame(read_hdf(filename, 'Data_Gold'))


def prepare_data(df: DataFrame) -> DataFrame:
    for c in df.columns:
        df[c + '_ret'] = df[c].pct_change().fillna(0)
    return df


def scale_data(data: DataFrame) -> (DataFrame, StandardScaler):
    df = data.copy()
    sc = StandardScaler()
    df.loc[:, 'Gold'] = sc.fit_transform(
        df.loc[:, 'Gold'].values.reshape(-1, 1))
    sc1 = StandardScaler()
    df.loc[:, 'Inflation'] = sc1.fit_transform(
        df.loc[:, 'Inflation'].values.reshape(-1, 1))
    df.loc[:, 'InterestRate'] = sc1.fit_transform(
        df.loc[:, 'InterestRate'].values.reshape(-1, 1))
    df.loc[:, 'DJI'] = sc1.fit_transform(
        df.loc[:, 'DJI'].values.reshape(-1, 1))
    return df, sc


def scale_back(data, custom_scaler, save=False):
    if save is True:
        data.to_hdf('DeepLearning_withPred.h5', 'Pred_LSTM')
    df = data.copy()
    df.loc[:, 'Gold'] = custom_scaler.inverse_transform(
        df.loc[:, 'Gold'].values.reshape(-1, 1))
    df.loc[:, 'Pred'] = custom_scaler.inverse_transform(
        df.loc[:, 'Pred'])
    return df


def training_data(time_series):
    X = np.atleast_3d(
        np.array([time_series[start:start + look_back] for start in
                  range(0, time_series.shape[0] - look_back)]))
    y = time_series[look_back:]
    return X, y


def prepare_input(split_date='2015-01-01'):
    raw_data = read_data()
    print('Raw data shape: {}'.format(raw_data.shape))

    scaled_data, scaler = scale_data(prepare_data(raw_data))
    train = scaled_data.loc[scaled_data.index < to_datetime(split_date)]
    test = scaled_data.loc[scaled_data.index >= to_datetime(split_date)]

    # train_data = raw_data.loc[raw_data.index < to_datetime(split_date)]
    # train, scaler_train = scale_data(prepare_data(train_data))
    # test_data = raw_data.loc[raw_data.index >= to_datetime(split_date)]
    # test, scaler_test = scale_data(prepare_data(test_data))

    print('Train data shape: {}'.format(train.shape))
    print('Test data shape: {}'.format(test.shape))
    return train, test, scaler


def split_input(train, test, target, predictors):
    # train_X = np.atleast_2d(train.loc[:, predictors].values)
    train_series = np.atleast_2d(train.loc[:, predictors])
    train_X = np.atleast_3d(
        np.array([train_series[start:start + look_back] for start in
                  range(0, train_series.shape[0] - look_back)]))
    train_y = train.loc[:, target].values[look_back:]

    # test_X = test.loc[:, predictors].values
    test_series = np.atleast_2d(test.loc[:, predictors])
    test_X = np.atleast_3d(
        np.array([test_series[start:start + look_back] for start in
                  range(0, test_series.shape[0] - look_back)]))
    test_y = test.loc[:, target].values[look_back:]

    # reshape input to be 3D [samples, timesteps, features]
    # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    print('Train_X shape:', train_X.shape)
    print('Train_y shape:', train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print('Train_X shape:', test_X.shape)
    print('Test_y shape:', test_y.shape)

    return train_X, train_y, test_X, test_y


def model_build(timesteps, features):
    custom_model = Sequential()
    custom_model.add(
        LSTM(lstm_units,
             input_shape=(timesteps, features),
             # output_dim=lstm_output_dim,
             return_sequences=True))
    custom_model.add(
        LSTM(lstm_units,
             input_shape=(timesteps, features),
             # output_dim=lstm_output_dim,
             return_sequences=False))
    custom_model.add(Dense(1))
    custom_model.add(Activation('linear'))
    custom_model.compile(loss="mse", optimizer="rmsprop")
    return custom_model


def train_model(train_X, train_y, plot=False):
    model_ = model_build(train_X.shape[1], train_X.shape[2])
    history = model_.fit(train_X, train_y,
                         epochs=lstm_epochs,
                         batch_size=80,
                         verbose=4,
                         shuffle=False)
    if plot is True:
        plt.plot(history.history['loss'], label='train')
        plt.legend()
        plt.show()
    return model_


def single_prediction(model, previous, predicted, predictors):
    if previous.empty is False:
        a = np.append(previous.Gold.values, predicted)
    else:
        a = np.array(predicted)
    y_pred = model.predict(a.reshape((1, look_back * len(predictors), 1)))
    return y_pred[0][0]


# make one forecast with an LSTM,
def forecast_lstm(model, X):
    # reshape input pattern to [samples, timesteps, features]
    # X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X)
    # convert to array
    return forecast[0][0]


# evaluate the persistence model
def make_forecasts(model, test_X, test_y, num_predictors, plot=True):
    forecasts = list()
    for i in range(len(test_X)):
        forecast = forecast_lstm(
            model, test_X[i].reshape(1, look_back, num_predictors))
        forecasts.append(forecast)

    if plot is True:
        plt.plot(test_y, 'g-')
        plt.plot(forecasts, 'b-')
        plt.grid()
        plt.show()
    return forecasts


def main(target,
         predictors,
         split_date,
         predict_training=False,
         save_forecast=False):
    # Prepare the datasets
    train, test, scaler = prepare_input(split_date)
    train_X, train_y, test_X, test_y = split_input(
        train, test, target, predictors)

    # Train the model
    model = train_model(train_X, train_y)

    # Test on training or test set?
    if predict_training is True:
        response = train_y
        forecast = make_forecasts(model, train_X, train_y, len(predictors))
    else:
        response = test_y
        forecast = make_forecasts(model, test_X, test_y, len(predictors))

    # Save results?
    if save_forecast is True:
        filename = 'forecast_{}.csv'.format('_'.join(predictors))
        result = DataFrame(
            {'test_y': scaler.inverse_transform(response),
             'forecast': scaler.inverse_transform(forecast)})
        result.to_csv(filename, index=False)
        print('Saved forecast to file: {}'.format(filename))


look_back: int = 12
lstm_units: int = 128
lstm_epochs: int = 100
my_target = 'Gold'
my_predictors = ['Gold', 'Inflation']  # , 'DJI', 'InterestRate']

main(my_target,
     my_predictors,
     split_date='2017-02-01',
     predict_training=True,
     save_forecast=True)
