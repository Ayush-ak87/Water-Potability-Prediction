import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from dvclive import Live
import yaml
import dagshub
import os
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifact
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models import infer_signature

#Load DagsHub Token from the environment variables
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Initialize DagsHub for experiment tracking
dagshub_url = "https://dagshub.com"
repo_owner='Ayush-ak87' 
repo_name='Water-Quality-Prediction'
mlflow.set_experiment("Final Model ")
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error Loading data from {filepath}: {e}")

# test_data = pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        x = data.drop(columns=["Potability"], axis=1)
        y = data["Potability"]
        return x,y
    except Exception as e:
        raise Exception(f"Error Preparing Data: {e}")

# x_test = test_data.iloc[:,0:-1].values
# y_test = test_data.iloc[:,-1].values

def load_model(filepath: str):
    try:
        with open(filepath,"rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")

# model = pickle.load(open("model.pkl", "rb"))

def evaluation_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml","r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        y_pred = model.predict(x_test)

        # Calculate Metrics
        acc = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall  =recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)

        # Log Params to MLFlow
        mlflow.log_param("Test_size",test_size)
        mlflow.log_param("n_estimators",n_estimators)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix Plot")
        cm_path = f"confusion_matrix.png"
        plt.savefig(cm_path)

        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)

        metrics_dict = {

            'accuracy':acc,
            'precision':precision,
            'recall':recall,
            'f1_score':f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error Evaluating Model: {e}")

def save_metrics(metrics_dict:dict, filepath: str) -> None:
    try:
        with open("reports/metrics.json","w") as file:
            json.dump(metrics_dict,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}")
    
def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        test_data = load_data(test_data_path)
        x_test,y_test = prepare_data(test_data)
        model = load_model(model_path)

        # Start MLflow run
        with mlflow.start_run() as run:
            metrics = evaluation_model(model, x_test, y_test)
            save_metrics(metrics,metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            
            # Log the source code file
            mlflow.log_artifact(__file__)

            signature = infer_signature(x_test,model.predict(x_test))

            mlflow.sklearn.log_model(model,"Best Model",signature=signature)

            #Save run ID and model info to JSON File
            run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)

    except Exception as e:
        raise Exception(f"An Error ocuured: {e}")
    
if __name__ =="__main__":
    main()