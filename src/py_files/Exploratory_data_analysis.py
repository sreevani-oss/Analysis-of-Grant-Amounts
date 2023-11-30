#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def summary_statistics(self) -> pd.DataFrame:
        """Returns summary statistics for numeric attributes."""
        return self.data.describe()
        
    def clean_dataset(self) -> pd.DataFrame:
        # Calculate the mean for each numeric column
        numeric_columns = self.data.select_dtypes(include=[np.number])
        column_means = numeric_columns.mean()
        
        # Replace NaN values with the corresponding column mean
        self.data[numeric_columns.columns] = self.data[numeric_columns.columns].fillna(column_means)
        
        return self.data

    def plot_time_series_trend(self, country_name, indicator_name):
        # Filtering the data for the specified country and indicator
        df = self.data.copy()
        df_melted = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')

        country_data = df_melted[(df_melted['Country Name'] == country_name) & (df_melted['Indicator Name'] == indicator_name)]     
        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(country_data['Year'], country_data['Value'])
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.title(f'Trend Over the Years for {country_name} ({indicator_name})')
        plt.grid(True)
        plt.show()

    def meanplot(self):
        df_melted = self.data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')
        year_mean = df_melted.groupby('Year')['Value'].mean().reset_index()
        # Create a bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Year', y='Value', data=year_mean)
        plt.xlabel('Year')
        plt.ylabel('Mean Value')
        plt.title('Mean Value by Year')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()

      
        






# In[ ]:




