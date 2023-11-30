#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

class DataSummary:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def dataset_shape(self) -> tuple:
        """Returns the shape of the dataset."""
        return self.data.shape

    def attribute_data_types(self) -> pd.Series:
        """Returns the data types for each attribute."""
        return self.data.dtypes

    def missing_values_count(self) -> pd.Series:
        """Counts the missing values for each attribute."""
        return self.data.isnull().sum()

    def categorical_data_analysis(self):
        """Explore categorical variables in the DataFrame."""
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        print("\nCategorical Columns:", categorical_columns)
        for col in categorical_columns:
            print(f"\nColumn: {col}\n", self.data[col].value_counts())



# In[ ]:




