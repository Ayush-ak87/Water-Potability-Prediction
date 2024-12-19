import numpy as np
import pandas as pd
import pickle
import json
from dvclive import Live
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

        acc = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall  =recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy",acc)
            live.log_metric("precision",precision)
            live.log_metric("recall",recall)
            live.log_metric("f1_score",f1)

            live.log_param("test_size",test_size)
            live.log_param("n_estimators",n_estimators)

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
        metrics = evaluation_model(model, x_test, y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An Error ocuured: {e}")
    
if __name__ =="__main__":
    main()