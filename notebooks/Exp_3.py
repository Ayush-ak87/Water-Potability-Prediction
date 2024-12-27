import pandas as pd
import numpy as np
import mlflow
import kagglehub
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub
from sklearn.model_selection import train_test_split

# Initialize DagsHub and setup MLFlow for Experiment Tracking
dagshub.init(repo_owner='Ayush-ak87', repo_name='Water-Quality-Prediction', mlflow=True)
mlflow.set_experiment("Water_Exp_3")
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
def fill_missing_with_mean(df):
  for column in df.columns:
    if df[column].isna().any():
      mean_value = df[column].mean()
      df[column].fillna(mean_value, inplace=True)
  return df

#fill missing values in train and test data
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# Split the data into features (x) and target (y) for training and testing
x_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values
x_test = test_processed_data.iloc[:,0:-1].values
y_test = test_processed_data.iloc[:,-1].values

# Define multiple baseline models to compare performance
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XG Boost": XGBClassifier()
}

with mlflow.start_run(run_name="Water Potability Models Experiment"):
    # Iterate over each model in the dictionary
    for model_name, model in models.items():
        # Start a child run within the parent run for each individual model
        with mlflow.start_run(run_name=model_name, nested=True):
            # Train the model on the training data
            model.fit(x_train, y_train)

           # Save the trained model using pickle
            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_filename, "wb"))
            
            # Make predictions on the test data
            y_pred = model.predict(x_test)
            
            # Calculate performance metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log the calculated metrics to MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Generate and visualize the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Create a heatmap of the confusion matrix
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png")  # Save the plot
            
            # Log the confusion matrix plot as an artifact in MLflow
            mlflow.log_artifact(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
            
            # Log the model to MLflow
            mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))
    
    # Log the source code file for reproducibility
    mlflow.log_artifact(__file__)

    # Set tags for the run to provide additional metadata
    mlflow.set_tag("author", "Ayush Kumar")

    # Preparing the Dataset for Logging
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    # Log the Train and Test Data in MLFlow
    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")
    
    print("All models have been trained and logged as child runs successfully.")