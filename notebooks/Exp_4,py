import numpy as np
import pandas as pd
import os
import mlflow
import kagglehub
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import pickle
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize DagsHub and setup MLFlow for Experiment Tracking
dagshub.init(repo_owner='Ayush-ak87', repo_name='Water-Quality-Prediction', mlflow=True)
mlflow.set_experiment("Water_Exp_4")
mlflow.set_tracking_uri("https://dagshub.com/Ayush-ak87/Water-Quality-Prediction.mlflow")

# Download latest version
path = kagglehub.dataset_download("uom190346a/water-quality-and-potability")

print("Path to dataset files:", path)

# Locate and read the CSV file
for file_name in os.listdir(path):
    if file_name.endswith(".csv"):  # Check if the file is a CSV
        csv_path = os.path.join(path, file_name)
        print("Reading CSV file:", csv_path)
        df = pd.read_csv(csv_path)  # Read the CSV file into a pandas DataFrame
        print(df.head())  # Display the first few rows

#Splitting Dataset into train set and test set
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

#Fill missing value with median
def fill_missing_with_median(df):
  for column in df.columns:
    if df[column].isna().any():
      median = df[column].median()
      df[column].fillna(median, inplace=True)
  return df

#fill missing values in train and test data
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Split the data into features (x) and target (y) for training and testing
x_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

# Define the Random Forest Classifier model and the parameter distribution for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],  # Different values of n_estimators to try
    'max_depth': [None, 4, 5, 6, 10],  # Different max_depth values to explore
}

# Perform RandomizedSearchCV to find the best hyperparameters for the Random Forest model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)


with mlflow.start_run(run_name="Random Forest Tuning ") as parent_run:
    random_search.fit(x_train,y_train)

    # Fit the RandomizedSearchCV object on the training data to identify the best hyperparameters
    random_search.fit(x_train, y_train)

     # Log the parameters and mean test scores for each combination tried
    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])  # Log the parameters
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])  # Log the mean test score

    # Print the best parameter found by RandomizedSearchCV
    print("Best Parameter Found: ",random_search.best_params_)

    # Log the best parameters in MLflow
    mlflow.log_params(random_search.best_params_)

    # Train the Model with the best parameters
    best_rf = random_search.best_estimator_
    best_rf.fit(x_train,y_train)

    # Save the trained model to a file
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Make predictions on the test set
    x_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values

    # Load the saved model
    model = pickle.load(open('model.pkl', "rb"))

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluation from test data
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log performance metrics into MLflow for tracking
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)

    # Log the training and testing data as inputs in MLflow
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    
    mlflow.log_input(train_df, "train")  # Log training data
    mlflow.log_input(test_df, "test")  # Log test data

    # Log the current script file as an artifact in MLflow
    mlflow.log_artifact(__file__)

    # Infer the model signature using the test features and predictions
    sign = infer_signature(x_test, random_search.best_estimator_.predict(x_test))
    
    # Log the trained model in MLflow with its signature
    mlflow.sklearn.log_model(random_search.best_estimator_, "Best Model", signature=sign)

    # Print the calculated performance metrics to the console for review
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)