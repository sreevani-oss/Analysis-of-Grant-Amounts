#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

class DataSummary:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    """
    Returns the shape of the dataset.

    Computes and returns the number of rows and columns in the DataFrame.

    Returns:
    - tuple: A tuple representing the shape of the DataFrame, with the first element being the number of rows and the second the number of columns.
    """
    def dataset_shape(self) -> tuple:
        """Returns the shape of the dataset."""
        return self.data.shape

    """
    Returns the data types for each attribute in the dataset.

    Computes and returns a Series with the data types of each column in the DataFrame.

    Returns:
    - pd.Series: A Series where the index represents the column names and the values are the data types of those columns.
    """
    def attribute_data_types(self) -> pd.Series:
        """Returns the data types for each attribute."""
        return self.data.dtypes

    """
    Counts the missing values for each attribute in the dataset.

    Computes and returns a Series with the count of missing (NaN) values for each column in the DataFrame.

    Returns:
    - pd.Series: A Series where the index represents the column names and the values are the counts of missing values in those columns.
    """
    def missing_values_count(self) -> pd.Series:
        """Counts the missing values for each attribute."""
        return self.data.isnull().sum()

    """
    Explores categorical variables in the DataFrame.

    Identifies all columns in the DataFrame that are of categorical data types (object or category) and prints the value counts for each of these columns.

    This method does not return a value but directly prints the results for each categorical column.
    """

    def categorical_data_analysis(self):
        """Explore categorical variables in the DataFrame."""
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        print("\nCategorical Columns:", categorical_columns)
        for col in categorical_columns:
            print(f"\nColumn: {col}\n", self.data[col].value_counts())



# In[ ]:




