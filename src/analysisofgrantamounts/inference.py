#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

class Inference:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = LinearRegression()



    # Data Preparation Function
    def prepare_data(self, country_name, window_size):
        """
        Prepares the data for training the machine learning model.
        """
        country_data = self.data[self.data['Country Name'] == country_name].iloc[:, 4:]
        X = []
        y = []
        for i in range(window_size, len(country_data.columns) - 1):
            X.append(country_data.iloc[0, i - window_size:i].values)
            y.append(country_data.iloc[0, i])
        return np.array(X), np.array(y)


    def train_model(self, X_train, y_train):
        """
        Trains the Linear Regression model using the class attribute.
        """
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def make_predictions(self,model, X_test):
        """
        Makes predictions using the trained model.
        """
        
        return model.predict(X_test)

    # Evaluation Function
    def evaluate_model(self, y_test, y_pred):
        """
        Evaluates the model's performance.
        """
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return rmse

    # Plotting Function
    def plot_predictions(self, y_test, y_pred):
        """
        Plots the actual vs predicted values.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='Actual', marker='o')
        plt.plot(y_pred, label='Predicted', marker='x')
        plt.title('Actual vs Predicted Grants')
        plt.xlabel('Year')
        plt.ylabel('Grants (current US$)')
        plt.legend()
        plt.show()



    def preprocess_data(self):
        """
        Preprocesses the data by merging historical grant data with economic indicators.

        :return: A DataFrame containing the merged data ready for model training.
        """
        # Assuming the two DataFrames can be joined on a common 'Year' column
        combined_data = pd.merge(self.historical_data, self.economic_indicators, on='Year')
        # Handling missing values if necessary
        combined_data.fillna(method='ffill', inplace=True)  # Forward fill for time series data
        return combined_data

    def train_test_split(self, data, test_size=1):
        """
        Splits the data into training and testing sets.

        :param data: The preprocessed DataFrame.
        :param test_size: The number of years to withhold for the test set.
        :return: Four DataFrames corresponding to X_train, X_test, y_train, y_test.
        """
        # Splitting data - using all but the last 'test_size' years for training
        X = data.drop('GrantAmount', axis=1)  # Assuming 'GrantAmount' is the target variable
        y = data['GrantAmount']
        X_train = X[:-test_size]
        X_test = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Trains a Linear Regression model.

        :param X_train: Training features.
        :param y_train: Training target.
        :return: A trained Linear Regression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def make_predictions(self, model, X_test):
        """
        Makes predictions using the trained model.

        :param model: The trained Linear Regression model.
        :param X_test: Testing features.
        :return: Predictions for the test set.
        """
        return model.predict(X_test)

    

    




