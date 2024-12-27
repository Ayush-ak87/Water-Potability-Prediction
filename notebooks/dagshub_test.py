import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/Ayush-ak87/Water-Quality-Prediction.mlflow")

dagshub.init(repo_owner='Ayush-ak87', repo_name='Water-Quality-Prediction', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)