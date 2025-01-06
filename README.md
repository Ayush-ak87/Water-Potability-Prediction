# Water Potability Prediction with MLOps

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Ayush-ak87/Water-Quality-Prediction.git)

## Overview
This project demonstrates a complete **MLOps workflow** for predicting water potability based on water quality metrics. It combines cutting-edge tools and methodologies like **MLflow**, **DVC**, **FastAPI**, and **Docker** to create a robust, scalable, and automated pipeline for machine learning model development, deployment, and monitoring.

The project also includes a **Tkinter-based desktop application** and a **FastAPI-based web interface** for user-friendly predictions.

## Objectives
- **Predict water potability**: Develop an ML model that classifies water as potable or non-potable based on its quality metrics.
- **Streamline the MLOps process**: Utilize modern tools for experiment tracking, data versioning, continuous integration, and deployment.
- **Deliver practical usability**: Provide both desktop and web-based applications for real-world utility.

---

## Features
- **End-to-End MLOps Workflow**:
  - Experiment tracking with **MLflow** and **DagsHub**.
  - Data versioning using **DVC**.
  - Continuous Integration (CI) with **GitHub Actions**.
  - Model containerization and deployment using **Docker**.
- **Interactive Applications**:
  - A desktop application built with **Tkinter**.
  - A web-based application developed using **FastAPI**.
- **Robust ML Pipeline**:
  - Data preprocessing and feature engineering.
  - Hyperparameter tuning for optimal performance.
  - Model registry for streamlined deployment.

---

## Project Workflow

### 1. **Setup**
- Utilize a pre-configured **Cookiecutter template** to set up a structured project directory.
- Initialize version control with Git and push the repository to GitHub.

### 2. **Experiment Tracking**
- Use **MLflow** integrated with **DagsHub** for experiment logging and artifact tracking.
- Experimentation:
  - **Baseline Model**: Random Forest with default parameters.
  - **Advanced Models**: Logistic Regression, XGBoost.
  - **Imputation Testing**: Comparison of mean vs. median strategies for missing values.
  - **Hyperparameter Tuning**: Optimize Random Forest for best performance.

### 3. **DVC Pipeline**
- Implement a robust pipeline with the following stages:
  - **Data Collection**: Organize and load data.
  - **Data Preprocessing**: Impute missing values and transform features.
  - **Model Training**: Train and evaluate models using cross-validation.
  - **Pipeline Execution**: Automate the workflow using DVC.

### 4. **Model Registration**
- Register the best-performing model in the **MLflow Model Registry**.
- Include metadata and performance metrics for easy tracking and retrieval.

### 5. **Applications**
- **Desktop Application**:
  - Built using **Tkinter**.
  - Fetches the latest registered model from MLflow.
  - Allows users to input water quality metrics and get predictions.
- **Web Application**:
  - Developed using **FastAPI** for a modern, interactive user interface.
  - Supports real-time predictions via a **FastAPI**.

### 6. **Continuous Integration (CI) Pipeline**
- Automate the following steps using **GitHub Actions**:
  - Install dependencies and run the DVC pipeline.
  - Test the model and promote it to production.
  - Build a Docker image and push it to DockerHub.

### 7. **Dockerization**
- Containerize the entire application with **Docker** for portability and ease of deployment.
- Publish the Docker image to DockerHub for reuse and scalability.

---

## Results and Analysis
- **Best Model**: Random Forest with mean imputation.
- **Optimal Hyperparameters**: `n_estimators=1000`, `max_depth=None`.
- **Performance Metrics**: Achieved high accuracy and F1-score on validation data.

---

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayush-ak87/Water-Quality-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Water-Quality-Prediction
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline:
   ```bash
   dvc repro
   ```

---

## Usage
- **Desktop Application**:
  Run the Tkinter app:
  ```bash
  python src/app/tkinter_app.py
  ```
- **Web Application**:
  Start the FastAPI server:
  ```bash
  uvicorn src.app.fastapi_app:app --reload
  ```
  Access the API at `http://localhost:8000`.

---

## Contributions
Contributions are welcome! Submit issues or pull requests to improve the project.

---

## License
This project is licensed under the [MIT License](LICENSE).

---
