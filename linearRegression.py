# data
import pandas as pd
import numpy as np

# machine learning
import keras
import ml_edu.experiment
import ml_edu.results

# data visualization
import plotly.express as px

chicago_taxi_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# updates daataframe to use specific columns.
training_df = chicago_taxi_dataframe.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]
training_df.head(200)
# print(training_df.describe(include='all'))

# What is the maximum fare?
# max_fare = training_df['FARE'].max()
max_fare = training_df['FARE'].describe()['max']
print(f"What is the maximum fare?  Answer: ${max_fare:.2f}")


# What is the mean distance across all trips?
# mean_distance = training_df['TRIP_MILES'].mean()
mean_distance = training_df['TRIP_MILES'].describe()['mean']
print(f"What is the mean distance across all trips? Answer: {mean_distance:.4f}")

# How many cabs companies are in the dataset?
# num_unique_companies = len(training_df['COMPANY'].unique())
num_unique_companies = training_df['COMPANY'].describe()['unique']
print(f"How many cabs companies are in the dataset? Answer: {num_unique_companies}")


# What is the most frequent payment type?
# most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
most_freq_payment_type = training_df['PAYMENT_TYPE'].describe()['top']
print(f"What is the most frequent payment type? Answer: {most_freq_payment_type}")

# Are any feature missing data?
missing_values = training_df.isnull().sum().sum()
print(f"Are any feature missing data? Answer:", "No" if missing_values == 0 else "Yes")

# print(training_df.describe(include='all'))
print(training_df.corr(numeric_only=True))

# View pairplot
fig = px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
fig.show()