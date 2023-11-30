#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

class Inference:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def top_n_countries(self, year: str, n: int) -> pd.DataFrame:
        """Identifies top N countries/entities based on grant amounts for a given year."""
        return self.data[['Country Name', year]].nlargest(n, year)

    def plot_grant_trends(self, countries: list):
        """Plots grant trends for given countries/entities over the years."""
        trend_data = self.data[self.data['Country Name'].isin(countries)].set_index('Country Name').iloc[:, 3:-1].transpose()
        plt.figure(figsize=(16, 8))
        for country in trend_data.columns:
            plt.plot(trend_data.index, trend_data[country], label=country, marker='o')
        plt.title('Grant Trends for Selected Entities')
        plt.xlabel('Year')
        plt.ylabel('Grant Amounts')
        plt.legend()
        plt.grid(axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def global_grant_trend(self):
        """Analyzes and plots the global trend of grant allocations over the years."""
        # Summing grant amounts for each year
        yearly_totals = self.data.iloc[:, 3:].sum()
    
        # Converting the index to a list of years (as strings)
        years = list(yearly_totals.index.astype(str))
        total_grants = yearly_totals.values
    
        # Plotting
        plt.figure(figsize=(16, 8))
        plt.plot(years, total_grants, marker='o')
        plt.title('Global Trend of Grant Allocations Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Total Grant Amount')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()



    def forecast_grants(self, forecast_years=5):
        # Load and process the dataset
        #print(self.data)
        #self.data = self.data.fillna(self.data.mean())
        self.data = self.data.dropna(subset=['Country Name', 'Country Code'])
        year_columns = self.data.columns[4:]
        total_grants_by_year = self.data[year_columns].sum()

        # Check for stationarity and make the series stationary if needed
        result = adfuller(total_grants_by_year.dropna())
        p_value = result[1]
        if p_value > 0.05:
            # Differencing the series to make it stationary
            total_grants_by_year_diff = total_grants_by_year.diff().dropna()
        else:
            total_grants_by_year_diff = total_grants_by_year

        # Fit the ARIMA model
        model = ARIMA(total_grants_by_year_diff, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast the specified number of years
        forecast = model_fit.forecast(steps=forecast_years)

        # Convert the differenced forecast back to the original scale
        last_value = total_grants_by_year.iloc[-1]
        forecast_cumsum = forecast.cumsum()
        forecast_original_scale = last_value + forecast_cumsum

        # Set the index for the forecasted years
        last_year = int(total_grants_by_year.index[-1])
        forecast_original_scale.index = np.arange(last_year + 1, last_year + 1 + forecast_years)

        # Return the forecasted values
        return forecast_original_scale


    def plot_grant_allocations_over_time(self):
        """
        Plots the total grant allocations over time.

        Parameters:
        - df (DataFrame): The DataFrame containing grant data with years as columns.
        """
        # Extracting year columns (all columns after the first four are years)
        year_columns = self.data.columns[4:]

        # Summing up the grant values for each year
        total_grants_by_year = self.data[year_columns].sum()
        # Plotting the results
        plt.figure(figsize=(15, 6))
        total_grants_by_year.plot(kind='line', color='blue', marker='o')
        plt.title('Global Grant Allocations Over Time')
        plt.xlabel('Year')
        plt.ylabel('Total Grants (in USD)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def analyze_grant_allocations(self, top_n=10):
        """
        Analyzes and plots grant allocations by country and year-wise data for top countries.

        Parameters:
        - df (DataFrame): The DataFrame containing grant data.
        - top_n (int): The number of top countries or regions to analyze.
        """
        # Summing up the grant values for each country over all years
        total_grants_by_country = self.data.set_index(['Country Name']).iloc[:, 3:].sum(axis=1)

        # Sorting the results to find the top countries or regions
        sorted_total_grants_by_country = total_grants_by_country.sort_values(ascending=False)

        # Identifying countries or regions with the most grants
        top_countries_or_regions = sorted_total_grants_by_country.head(top_n)

        # Plotting the top countries or regions
        plt.figure(figsize=(10, 6))
        top_countries_or_regions.plot(kind='bar', color='green')
        plt.title('Top Countries or Regions by Total Grants Received')
        plt.xlabel('Country or Region')
        plt.ylabel('Total Grants (in USD)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Extracting the top few countries for a more detailed year-wise analysis
        top_countries_list = top_countries_or_regions.index.tolist()
        yearly_grants_top_countries = self.data[self.data['Country Name'].isin(top_countries_list)]

        # Extracting year columns
        year_columns = self.data.columns[4:]

        # Plotting year-wise data for these countries
        plt.figure(figsize=(15, 6))
        for country in top_countries_list:
            yearly_data = yearly_grants_top_countries[yearly_grants_top_countries['Country Name'] == country]
            plt.plot(year_columns, yearly_data.iloc[0, 4:], label=country)

        plt.title('Yearly Grant Allocations for Top Countries/Regions')
        plt.xlabel('Year')
        plt.ylabel('Grants (in USD)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Example usage:
    # analyze_grant_allocations(your_dataframe, top_n=10)


    def plot_grants_by_region(self):
        """
        Plots the total grant allocations by region using the provided DataFrame.

        Parameters:
        - df (DataFrame): The DataFrame containing grant data with 'Country Name' as regions.
        """
        # Grouping data by regions and summing up the grants for each region
        region_wise_grants = self.data.groupby('Country Name').sum().iloc[:, 3:]

        # Calculating the total grants received by each region over all years
        total_grants_by_region = region_wise_grants.sum(axis=1).sort_values(ascending=False)

        # Plotting the total grants for each region using a horizontal bar chart
        plt.figure(figsize=(10, 15))
        total_grants_by_region.plot(kind='barh', color='teal')
        plt.title('Total Grant Allocations by Region')
        plt.ylabel('Region')
        plt.xlabel('Total Grants (in USD)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Example usage:
    # plot_grants_by_region(your_dataframe)





