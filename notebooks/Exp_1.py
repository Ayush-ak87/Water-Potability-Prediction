import numpy as np
import pandas as pd
import os
import mlflow
import kagglehub
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# Initialize DagsHub and setup MLFlow for Experiment Tracking
dagshub.init(repo_owner='Ayush-ak87', repo_name='Water-Quality-Prediction', mlflow=True)
mlflow.set_experiment("Water_Exp_1")
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
from sklearn.model_selection import train_test_split
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
from sklearn.ensemble import RandomForestClassifier
x_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

n_estimators = 100

with mlflow.start_run():
    # Using Random Forest Classifier
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(x_train,y_train)

    #Save the model
    import pickle
    pickle.dump(model, open('model.pkl','wb'))

    # Prepare test data for prediction
    x_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values

    # Predict the target for the test data
    model = pickle.load(open('model.pkl','rb'))
    y_pred = model.predict(x_test)

    # Calculating the Performance Metrics for Evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics to MLFlow for Tracking
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    # Log the number of estimator used as a parameter
    mlflow.log_param("n_estimators",n_estimators)

    # Generate a Confusion Matrix to Visualize Model Performance
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    # Save Confusion Matrix Plot as PNG File
    plt.savefig("confusion_matrix.png")

    # Log the Confusion Matrix Plot to MLFlow
    mlflow.log_artifact("confusion_matrix.png")

    # Log the Trained Model To MLFlow
    mlflow.sklearn.log_model(model,"RandomForestClassifier")

    # Log the Source Code File in MLFlow for Reference
    mlflow.log_artifact(__file__)

    # Set tags in MLFlow to store additional Metadata
    mlflow.set_tag("author","Ayush Kumar")
    mlflow.set_tag("model","RF")

    # Preparing the Dataset for Logging
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    # Log the Train and Test Data in MLFlow
    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")

    # Printing the Performance Metrics for Evaluation
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)