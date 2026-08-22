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
# fig.show()

# Train Model
def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple linear regression model."""
    # Describe the topography of the model
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model topography into code that Keras can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
        loss="mean_squared_error",
        metrics=metrics
        )
    return model
def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
    ) -> ml_edu.experiment.Experiment:
    """Train the model by feeding it data"""
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(
        x=features,
        y=label,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs
        )
    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history=history),
    )
    
print("SUCCESS: defining linear regression function complete")
    
