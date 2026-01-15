# Bank Client Attrition Prediction

A machine learning project for predicting customer churn in the banking sector using PySpark MLlib, with an interactive Streamlit web application for real-time predictions.

## Overview

Customer attrition (churn) is a critical challenge for banks, where losing customers directly impacts revenue and growth. This project builds a predictive model to identify clients at high risk of leaving, enabling proactive retention strategies.

The project implements a complete ML pipeline including:
- Data preprocessing with outlier handling
- Class imbalance management using weighted classification
- Hyperparameter tuning with cross-validation
- Model evaluation with comprehensive metrics
- Real-time prediction interface via Streamlit

## Features

- **Big Data Processing**: Leverages Apache Spark for scalable data processing
- **Advanced ML Pipeline**: End-to-end pipeline with feature engineering, scaling, and classification
- **Class Imbalance Handling**: Implements weighted classification to handle imbalanced dataset (80/20 split)
- **Model Optimization**: Cross-validation with grid search for hyperparameter tuning
- **Data Persistence**: MongoDB integration for storing preprocessed data
- **Interactive Dashboard**: Streamlit application for real-time predictions with visual insights
- **Comprehensive Evaluation**: Multiple metrics (AUC-ROC, Accuracy, Precision, Recall, F1-Score)

## Tech Stack

- **Data Processing**: PySpark 4.0.1
- **Machine Learning**: PySpark MLlib (RandomForestClassifier)
- **Database**: MongoDB with Mongo Express UI
- **Web Application**: Streamlit
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Containerization**: Docker Compose
- **Environment**: Python 3.12

## Project Structure

```
Pr-diction-de-l-Attrition-Client-Bancaire/
├── attrition.ipynb          # Main Jupyter notebook with complete ML pipeline
├── app.py                   # Streamlit web application for predictions
├── dataset.csv              # Bank client data (10,000 records)
├── docker-compose.yml       # MongoDB and Mongo Express setup
├── model/                   # Saved trained model (gitignored)
│   └── best_rf_model/
├── attrition/              # Python virtual environment (gitignored)
├── test.ipynb              # Testing notebooks
├── test(1).ipynb
└── test.py
```

## Dataset

The dataset contains 10,000 bank client records with the following features:

**Features:**
- `CreditScore`: Client's credit score (350-850)
- `Geography`: Client's country (France, Spain, Germany)
- `Gender`: Male or Female
- `Age`: Client's age (18-92)
- `Tenure`: Years as a bank client (0-10)
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products used (1-4)
- `HasCrCard`: Whether the client has a credit card (0/1)
- `IsActiveMember`: Whether the client is an active member (0/1)
- `EstimatedSalary`: Client's estimated salary

**Target Variable:**
- `Exited`: Whether the client left the bank (0 = Stayed, 1 = Left)

**Class Distribution:**
- Class 0 (Stayed): 7,963 (79.6%)
- Class 1 (Left): 2,037 (20.4%)

## Installation

### Prerequisites

- Python 3.12+
- Docker and Docker Compose (for MongoDB)
- Java 8+ (required for PySpark)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Pr-diction-de-l-Attrition-Client-Bancaire
```

2. Create and activate virtual environment:
```bash
python -m venv attrition
source attrition/bin/activate  # On Linux/Mac
# or
attrition\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install pyspark pandas numpy matplotlib seaborn plotly streamlit pymongo scikit-learn jupyter
```

4. Start MongoDB (optional, for data persistence):
```bash
docker-compose up -d
```
Access Mongo Express UI at: http://localhost:8081

## Usage

### Training the Model

1. Open and run the Jupyter notebook:
```bash
jupyter notebook attrition.ipynb
```

2. Execute all cells to:
   - Load and explore the data
   - Preprocess and handle outliers
   - Encode categorical variables
   - Handle class imbalance
   - Train the model with cross-validation
   - Evaluate performance
   - Save the trained model

### Running the Web Application

1. Ensure the model is trained and saved in `model/best_rf_model/`

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Access the application at: http://localhost:8501

4. Use the sidebar to input client information and get real-time predictions

## Model Pipeline

The ML pipeline consists of three stages:

1. **VectorAssembler**: Combines all features into a single vector
2. **StandardScaler**: Normalizes features for better model performance
3. **RandomForestClassifier**: Trained with class weights to handle imbalance

### Hyperparameter Tuning

Cross-validation tested:
- `numTrees`: [50, 100]
- `maxDepth`: [5, 10, 15]
- `minInstancesPerNode`: [1, 5]

**Best Parameters:**
- numTrees: 50
- maxDepth: 10
- minInstancesPerNode: 1

## Model Performance

The final model achieves strong performance on the test set:

| Metric | Score |
|--------|-------|
| **AUC-ROC** | 0.8564 |
| **Accuracy** | 0.8272 |
| **Precision** | 0.8433 |
| **Recall** | 0.8272 |
| **F1-Score** | 0.8335 |

### Classification Report

```
              precision    recall  f1-score   support

   Resté (0)       0.88      0.89      0.88      1537
   Parti (1)       0.63      0.61      0.62       384

    accuracy                           0.83      1921
```

### Feature Importance

Top 5 most important features:
1. Age
2. NumOfProducts
3. Balance
4. Geography_Index
5. IsActiveMember

## Key Insights

1. **Geography Impact**: Germany has the highest attrition rate (32.4%) compared to France (16.2%) and Spain (16.7%)
2. **Gender Disparity**: Female clients have higher attrition (25.1%) than male clients (16.5%)
3. **Product Usage**: Clients with 3-4 products show significantly higher attrition rates
4. **Age Factor**: Clients who left are older on average (44.7 years) vs those who stayed (37.1 years)

## MongoDB Integration

Preprocessed data is stored in MongoDB for:
- Data versioning and tracking
- Easy access for other applications
- Historical analysis

**Connection Details:**
- Database: `bank_attrition`
- Collection: `preprocessed_data`
- Connection String: `mongodb://admin:admin123@localhost:27018/`

## Web Application Features

The Streamlit dashboard provides:
- Interactive input fields for client data
- Real-time prediction with probability scores
- Visual risk gauge with color-coded indicators
- Actionable recommendations based on predictions
- Model performance metrics display

## Requirements

Key dependencies:
```
pyspark>=4.0.1
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.13.0
plotly>=5.14.0
streamlit>=1.28.0
pymongo>=4.5.0
scikit-learn>=1.3.0
```