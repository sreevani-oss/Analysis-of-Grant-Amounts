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

         # Data Cleaning Function
    def clean_data(self, columns_to_remove, fill_missing):
        """
        Cleans the dataset by handling missing values and removing unnecessary columns.
        """
        # Remove specified columns
        if columns_to_remove:
            self.data = self.data.drop(columns=columns_to_remove, errors='ignore')
        # Fill missing values
        if fill_missing == 'mean':
            self.data = self.data.fillna(self.data.mean(numeric_only=True))
        elif fill_missing == 'median':
            self.data = self.data.fillna(self.data.median(numeric_only=True))
        elif fill_missing == 'zero':
            self.data = self.data.fillna(0)
        elif fill_missing is None:
            pass  # Do nothing if fill_missing is None
       
        return self.data

    def trend_analysis(self):
        """Plots the trend of grants over the years."""
        yearly_averages = self.data.iloc[:, 4:].mean()
        plt.plot(yearly_averages.index, yearly_averages.values)
        plt.title('Average Grants per Year')
        plt.xlabel('Year')
        plt.ylabel('Average Grants')
        plt.show()


    def top_bottom_countries(self, n=5):
        """Identifies top and bottom n countries based on total grants received and plots their grants."""
        total_grants = self.data.iloc[:, 4:].sum(axis=1)
        self.data['Total Grants'] = total_grants
        top_countries = self.data.nlargest(n, 'Total Grants')[['Country Name', 'Total Grants']]
        bottom_countries = self.data.nsmallest(n, 'Total Grants')[['Country Name', 'Total Grants']]

        # Plotting top countries
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Total Grants', y='Country Name', data=top_countries, palette='viridis')
        plt.title(f'Top {n} Countries in Total Grants Received')
        plt.xlabel('Total Grants')
        plt.ylabel('Country Name')

        # Plotting bottom countries
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Total Grants', y='Country Name', data=bottom_countries, palette='rocket')
        plt.title(f'Bottom {n} Countries in Total Grants Received')
        plt.xlabel('Total Grants')
        plt.ylabel('Country Name')

        return top_countries, bottom_countries

    def distribution_analysis(self, year):
        """Plots the distribution of grants for a specific year."""
        sns.histplot(self.data[year], kde=True)
        plt.title(f'Distribution of Grants in {year}')
        plt.xlabel('Grants')
        plt.ylabel('Frequency')
        plt.show()

    def correlation_analysis(self):
        """Displays a heatmap of the correlation matrix between years."""
        corr_matrix = self.data.iloc[:, 4:].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Grants Over Years')
        plt.show()
