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
print(training_df.describe(include='all'))



