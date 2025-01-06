# Water Potability Prediction with MLOps

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Ayush-ak87/Water-Quality-Prediction.git)

## Project Overview

This project develops an End-to-End MLOps Workflow for predicting water potability based on water quality metrics. It integrates multiple tools and technologies, ensuring a structured and efficient machine learning lifecycle. Key highlights include Experiment Tracking with MLflow, Data Versioning with DVC, CI/CD Pipelines with GitHub Actions, and Model Containerization with Docker. The project also includes both a **Tkinter-based desktop application** and a **FastAPI-based UI** for user-friendly predictions.

---

## Objective
- **Predict Water Potability**: Develop an ML model to classify water as potable or non-potable using quality metrics.
- **Build a robust MLOps pipeline**: Automate data processing, model training, and deployment while ensuring reproducibility and scalability.
- **Deliver user-friendly applications**: Provide desktop and web-based solutions for real-world usability.

---

## Features

1. **Comprehensive MLOps Workflow**:
   - Experiment tracking with **MLflow**.
   - Data versioning and pipeline automation using **DVC**.
   - Continuous Integration and Deployment (CI/CD) with **GitHub Actions**.
   - Model containerization using **Docker**.

2. **Interactive Applications**:
   - A **Tkinter desktop app** for offline predictions.
   - A **FastAPI-based web interface** for modern, scalable predictions.

3. **Optimized Machine Learning Model**:
   - Extensive experimentation to identify the best model and preprocessing strategies.
   - Hyperparameter tuning for optimal performance.

4. **Proof of Concept**:
   - Developed as an individual project to showcase end-to-end MLOps capabilities.
   - Overcame significant challenges and gained a deep understanding of concepts over a month-long development period.

---

## Workflow

### 1. Data Sourcing and Understanding
- **Dataset**: Sourced from Kaggle, containing water quality metrics like pH, Hardness, Solids, etc., with labels indicating potability.
- Performed Exploratory Data Analysis (EDA) to understand feature distributions and relationships.

### 2. Experimentation
- Conducted **5 experiments**:
  - **Baseline Model**: Random Forest with default parameters.
  - **Model Comparisons**: Evaluated Logistic Regression, XGBoost, Random Forest, Decision Tree, Support Vector Machine(SVM).
  - **Imputation Strategies**: Tested mean vs. median imputation for handling missing values.
  - **Hyperparameter Tuning**: Fine-tuned Random Forest with mean imputation for `n_estimators=1000` and `max_depth=None`.

### 3. MLOps Implementation
- **MLflow for Experiment Tracking**:
  - Logged model metrics (Accuracy, Precision, Recall, F1-Score) and parameters.
  - Tracked artifacts and registered the best-performing model.

- **DVC for Data Versioning**:
  - Versioned raw, processed, and intermediate datasets.
  - Automated data collection, data preprocessing, model building, model evaluation and model registration pipelines.

- **GitHub Actions for CI/CD**:
  - Automated pipeline execution, model testing, and Docker image deployment.

- **Docker for Containerization**:
  - Built a Docker image for the application.
  - Pushed the image to DockerHub for easy deployment.

### 4. Applications
- **Tkinter Desktop App**:
  - A simple, user-friendly GUI for offline predictions.
  - Automatically fetches the latest registered model from MLflow.

- **FastAPI Web App**:
  - Modern web interface for real-time predictions.
  - Deployed locally with the option to scale to cloud platforms in the future.

---

## Challenges Faced
- Handling missing data and testing different imputation strategies.
- Understanding and implementing advanced MLOps tools like MLflow and DVC.
- Ensuring a modular and maintainable codebase with proper exception handling.
- Completing the project within a month while deeply learning all relevant concepts.

---

## Results and Key Insights
- **Best Model**: Random Forest with mean imputation.
- **Optimal Hyperparameters**: `n_estimators=1000`, `max_depth=None`.
- **Performance Metrics**:
  - Accuracy: Achieved high accuracy on validation data.
  - Precision, Recall, and F1-Score tracked and optimized for balanced performance.

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- Git and DVC
- Docker (optional, for containerized deployment)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayush-ak87/Water-Quality-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Water-Quality-Prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the DVC pipeline:
   ```bash
   dvc repro
   ```
5. Launch the Tkinter app:
   ```bash
   python src/app/tkinter_app.py
   ```
6. Start the FastAPI server:
   ```bash
   uvicorn src.app.fastapi_app:app --reload
   ```

---

## Future Improvements
- Deploy the FastAPI app on a cloud platform like AWS, Azure, or GCP.
- Add real-time data fetching and processing capabilities.
- Incorporate advanced monitoring tools for model performance in production.
- Enhance the UI/UX of the applications.

---

## Contributions
This project is an individual effort, but contributions are welcome! Feel free to open issues or submit pull requests to improve the repository.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Kaggle**: For providing the dataset used in this project.
- **DataThinkers YouTube Channel**: For the tutorial videos that guided the development of this project and provided insights into implementing MLOps workflows.
- **Open-Source Tools and Libraries**: Gratitude to the developers of MLflow, DVC, Docker, FastAPI, and other tools that made this project possible.
