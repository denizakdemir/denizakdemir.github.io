#!/usr/bin/env python
"""
This script downloads and prepares three publicly available clinical datasets 
in the cardiology domain for use with a tabular autoencoder.
The datasets include:
  1. Framingham Heart Study dataset (from Kaggle)
  2. UCI Heart Disease (Cleveland) dataset (from the UCI repository)
  3. Kaggle Cardiovascular Disease dataset (~70K records)

For each dataset, the script:
  - Downloads the data (using requests for UCI or Kaggle API via os.system for Kaggle).
  - Processes the raw data into a CSV file.
  - Constructs a data dictionary CSV file that describes each variable 
    (including type, range or categorical mapping, and an overall description).

Ensure that you have:
  - Python 3.x installed.
  - The required packages: pandas, numpy, requests.
  - Kaggle API installed and configured (for Kaggle datasets).

Author: [Your Name]
Date: [Today's Date]
"""

import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import logging

# Configure logging for informative output.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def download_and_prepare_framingham():
    """
    Downloads and prepares the Framingham Heart Study dataset.
    
    This function uses the Kaggle API (via os.system call) to download the dataset,
    extracts the CSV, and creates a corresponding data dictionary CSV file.
    """
    logging.info("Downloading Framingham Heart Study dataset from Kaggle...")
    
    # Kaggle API command. (Ensure your Kaggle API is configured.)
    # The dataset slug used here is hypothetical. Replace with the correct slug if necessary.
    kaggle_cmd = "kaggle datasets download -d kraztk/framingham-heart-study -p ."
    os.system(kaggle_cmd)
    
    zip_path = "framingham-heart-study.zip"
    if not os.path.exists(zip_path):
        logging.error("Framingham dataset zip file not found. Check Kaggle API and dataset slug.")
        return
    # Extract the ZIP file.
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        logging.info(f"Extracting files: {file_list}")
        csv_file_name = None
        for file in file_list:
            if file.endswith('.csv'):
                csv_file_name = file
                z.extract(file, path=".")
                break
        if csv_file_name is None:
            logging.error("CSV file not found in the Framingham dataset zip.")
            return

    # Rename the extracted CSV to a standard name.
    framingham_csv = "framingham.csv"
    if csv_file_name != framingham_csv:
        os.rename(csv_file_name, framingham_csv)
    
    logging.info("Creating data dictionary for Framingham dataset...")
    # Construct a data dictionary.
    # Note: The following variable ranges and mappings are illustrative.
    dict_data = [
        {'name': 'age', 'type': 'integer', 'description': 'Age in years', 'min': 28, 'max': 62},
        {'name': 'male', 'type': 'categorical', 'description': 'Gender (1=male, 0=female)', 'value': '0', 'label': 'Female'},
        {'name': 'male', 'type': 'categorical', 'description': 'Gender (1=male, 0=female)', 'value': '1', 'label': 'Male'},
        {'name': 'education', 'type': 'integer', 'description': 'Education level (years)', 'min': 1, 'max': 20},
        {'name': 'currentSmoker', 'type': 'categorical', 'description': 'Smoking status (1=smoker, 0=non-smoker)', 'value': '0', 'label': 'Non-Smoker'},
        {'name': 'currentSmoker', 'type': 'categorical', 'description': 'Smoking status (1=smoker, 0=non-smoker)', 'value': '1', 'label': 'Smoker'},
        {'name': 'cigsPerDay', 'type': 'continuous', 'description': 'Number of cigarettes smoked per day', 'min': 0, 'max': 50},
        {'name': 'BPMeds', 'type': 'categorical', 'description': 'Blood pressure medication (1=yes, 0=no)', 'value': '0', 'label': 'No'},
        {'name': 'BPMeds', 'type': 'categorical', 'description': 'Blood pressure medication (1=yes, 0=no)', 'value': '1', 'label': 'Yes'},
        {'name': 'prevalentStroke', 'type': 'categorical', 'description': 'History of stroke (1=yes, 0=no)', 'value': '0', 'label': 'No'},
        {'name': 'prevalentStroke', 'type': 'categorical', 'description': 'History of stroke (1=yes, 0=no)', 'value': '1', 'label': 'Yes'},
        {'name': 'prevalentHyp', 'type': 'categorical', 'description': 'Hypertension (1=yes, 0=no)', 'value': '0', 'label': 'No'},
        {'name': 'prevalentHyp', 'type': 'categorical', 'description': 'Hypertension (1=yes, 0=no)', 'value': '1', 'label': 'Yes'},
        {'name': 'diabetes', 'type': 'categorical', 'description': 'Diabetes (1=yes, 0=no)', 'value': '0', 'label': 'No'},
        {'name': 'diabetes', 'type': 'categorical', 'description': 'Diabetes (1=yes, 0=no)', 'value': '1', 'label': 'Yes'},
        {'name': 'totChol', 'type': 'continuous', 'description': 'Total cholesterol (mg/dL)', 'min': 100, 'max': 400},
        {'name': 'sysBP', 'type': 'continuous', 'description': 'Systolic blood pressure (mm Hg)', 'min': 90, 'max': 200},
        {'name': 'diaBP', 'type': 'continuous', 'description': 'Diastolic blood pressure (mm Hg)', 'min': 60, 'max': 120},
        {'name': 'BMI', 'type': 'continuous', 'description': 'Body Mass Index (kg/m^2)', 'min': 18, 'max': 40},
        {'name': 'heartRate', 'type': 'continuous', 'description': 'Heart rate (beats per minute)', 'min': 50, 'max': 100},
        {'name': 'glucose', 'type': 'continuous', 'description': 'Glucose level (mg/dL)', 'min': 70, 'max': 200},
        {'name': 'target', 'type': 'categorical', 'description': '10-year CHD risk (0=No, 1=Yes)', 'value': '0', 'label': 'No CHD'},
        {'name': 'target', 'type': 'categorical', 'description': '10-year CHD risk (0=No, 1=Yes)', 'value': '1', 'label': 'CHD'},
        # Overall dataset description.
        {'name': 'dataset', 'type': 'overall', 'description': 'Framingham Heart Study dataset',
         'overall_description': 'This dataset contains clinical and lifestyle factors for predicting 10-year coronary heart disease risk.'}
    ]
    framingham_dict_df = pd.DataFrame(dict_data)
    framingham_dict_csv = "framingham_dict.csv"
    framingham_dict_df.to_csv(framingham_dict_csv, index=False)
    logging.info("Framingham dataset and dictionary saved.")


def download_and_prepare_uci_heart():
    """
    Downloads and prepares the UCI Heart Disease (Cleveland) dataset.
    
    This function downloads the processed Cleveland data from the UCI repository,
    assigns proper column names, converts missing values ('?') to NaN, 
    and creates a data dictionary CSV file.
    """
    logging.info("Downloading UCI Heart Disease dataset from UCI repository...")
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error("Failed to download UCI Heart Disease dataset.")
        return
    
    # Define column names as per UCI documentation.
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    # Read data into a DataFrame.
    data = pd.read_csv(io.StringIO(response.text), header=None, names=columns)
    
    # Replace '?' with np.nan for missing values.
    data.replace('?', np.nan, inplace=True)
    for col in ["ca", "thal"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    uci_csv = "data/uci_heart.csv"
    data.to_csv(uci_csv, index=False)
    logging.info("UCI Heart Disease data saved as uci_heart.csv")
    
    logging.info("Creating data dictionary for UCI Heart Disease dataset...")
    # Construct the data dictionary.
    dict_entries = [
        {'name': 'age', 'type': 'continuous', 'description': 'Age in years', 'min': data['age'].min(), 'max': data['age'].max()},
        {'name': 'sex', 'type': 'categorical', 'description': 'Sex (1=male; 0=female)', 'value': '0', 'label': 'Female'},
        {'name': 'sex', 'type': 'categorical', 'description': 'Sex (1=male; 0=female)', 'value': '1', 'label': 'Male'},
        {'name': 'cp', 'type': 'categorical', 'description': 'Chest pain type (1: typical angina, 2: atypical, 3: non-anginal, 4: asymptomatic)',
         'value': '1', 'label': 'typical angina'},
        {'name': 'cp', 'type': 'categorical', 'description': 'Chest pain type', 'value': '2', 'label': 'atypical'},
        {'name': 'cp', 'type': 'categorical', 'description': 'Chest pain type', 'value': '3', 'label': 'non-anginal'},
        {'name': 'cp', 'type': 'categorical', 'description': 'Chest pain type', 'value': '4', 'label': 'asymptomatic'},
        {'name': 'trestbps', 'type': 'continuous', 'description': 'Resting blood pressure (mm Hg)', 'min': data['trestbps'].min(), 'max': data['trestbps'].max()},
        {'name': 'chol', 'type': 'continuous', 'description': 'Serum cholesterol (mg/dL)', 'min': data['chol'].min(), 'max': data['chol'].max()},
        {'name': 'fbs', 'type': 'categorical', 'description': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)', 'value': '0', 'label': 'False'},
        {'name': 'fbs', 'type': 'categorical', 'description': 'Fasting blood sugar > 120 mg/dl', 'value': '1', 'label': 'True'},
        {'name': 'restecg', 'type': 'categorical', 'description': 'Resting ECG (0: normal, 1: ST-T abnormal, 2: LVH)',
         'value': '0', 'label': 'normal'},
        {'name': 'restecg', 'type': 'categorical', 'description': 'Resting ECG', 'value': '1', 'label': 'ST-T abnormal'},
        {'name': 'restecg', 'type': 'categorical', 'description': 'Resting ECG', 'value': '2', 'label': 'LVH'},
        {'name': 'thalach', 'type': 'continuous', 'description': 'Maximum heart rate achieved', 'min': data['thalach'].min(), 'max': data['thalach'].max()},
        {'name': 'exang', 'type': 'categorical', 'description': 'Exercise induced angina (1=yes, 0=no)', 'value': '0', 'label': 'No'},
        {'name': 'exang', 'type': 'categorical', 'description': 'Exercise induced angina', 'value': '1', 'label': 'Yes'},
        {'name': 'oldpeak', 'type': 'continuous', 'description': 'ST depression induced by exercise', 'min': data['oldpeak'].min(), 'max': data['oldpeak'].max()},
        {'name': 'slope', 'type': 'categorical', 'description': 'Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)',
         'value': '1', 'label': 'upsloping'},
        {'name': 'slope', 'type': 'categorical', 'description': 'Slope of the peak exercise ST segment', 'value': '2', 'label': 'flat'},
        {'name': 'slope', 'type': 'categorical', 'description': 'Slope of the peak exercise ST segment', 'value': '3', 'label': 'downsloping'},
        {'name': 'ca', 'type': 'continuous', 'description': 'Number of major vessels (0-3) colored by fluoroscopy', 'min': data['ca'].min(), 'max': data['ca'].max()},
        {'name': 'thal', 'type': 'categorical', 'description': 'Thalassemia (3=normal, 6=fixed, 7=reversible)', 'value': '3', 'label': 'normal'},
        {'name': 'thal', 'type': 'categorical', 'description': 'Thalassemia', 'value': '6', 'label': 'fixed defect'},
        {'name': 'thal', 'type': 'categorical', 'description': 'Thalassemia', 'value': '7', 'label': 'reversible defect'},
        {'name': 'num', 'type': 'categorical', 'description': 'Diagnosis of heart disease (0 = no disease, 1-4 = disease)',
         'value': '0', 'label': 'No disease'},
        {'name': 'num', 'type': 'categorical', 'description': 'Diagnosis of heart disease', 'value': '1', 'label': 'Disease'},
        # Overall description.
        {'name': 'dataset', 'type': 'overall', 'description': 'UCI Heart Disease (Cleveland) dataset',
         'overall_description': 'This dataset includes clinical features for heart disease diagnosis with both continuous and categorical variables, and contains missing values indicated by "?".'}
    ]
    uci_dict_df = pd.DataFrame(dict_entries)
    uci_dict_csv = "data/uci_heart_dict.csv"
    uci_dict_df.to_csv(uci_dict_csv, index=False)
    logging.info("UCI Heart Disease data dictionary saved.")


def download_and_prepare_cardio():
    """
    Downloads and prepares the Kaggle Cardiovascular Disease dataset.
    
    Uses the Kaggle API (via os.system call) to download the dataset,
    extracts the CSV file, optionally drops an 'id' column, and creates a data dictionary.
    """
    logging.info("Downloading Kaggle Cardiovascular Disease dataset...")
    kaggle_cmd = "kaggle datasets download -d sulianova/cardiovascular-disease-dataset -p ."
    os.system(kaggle_cmd)
    
    zip_path = "cardiovascular-disease-dataset.zip"
    if not os.path.exists(zip_path):
        logging.error("Cardiovascular dataset zip file not found. Check Kaggle API and dataset slug.")
        return
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        logging.info(f"Extracting files: {file_list}")
        csv_file_name = None
        for file in file_list:
            if file.endswith('.csv'):
                csv_file_name = file
                z.extract(file, path=".")
                break
        if csv_file_name is None:
            logging.error("CSV file not found in the Cardiovascular dataset zip.")
            return

    cardio_csv = "cardio.csv"
    if csv_file_name != cardio_csv:
        os.rename(csv_file_name, cardio_csv)
    
    # Load the dataset to check its columns.
    cardio_data = pd.read_csv(cardio_csv)
    # Drop the 'id' column if present.
    if 'id' in cardio_data.columns:
        cardio_data.drop('id', axis=1, inplace=True)
        cardio_data.to_csv(cardio_csv, index=False)
    
    logging.info("Creating data dictionary for Cardiovascular Disease dataset...")
    # Define the dictionary based on known columns.
    dict_entries = [
        {'name': 'age', 'type': 'continuous', 'description': 'Age in days (convert to years by dividing by 365)', 'min': cardio_data['age'].min(), 'max': cardio_data['age'].max()},
        {'name': 'gender', 'type': 'categorical', 'description': 'Gender (1 = male, 2 = female)', 'value': '1', 'label': 'Male'},
        {'name': 'gender', 'type': 'categorical', 'description': 'Gender (1 = male, 2 = female)', 'value': '2', 'label': 'Female'},
        {'name': 'height', 'type': 'continuous', 'description': 'Height in cm', 'min': cardio_data['height'].min(), 'max': cardio_data['height'].max()},
        {'name': 'weight', 'type': 'continuous', 'description': 'Weight in kg', 'min': cardio_data['weight'].min(), 'max': cardio_data['weight'].max()},
        {'name': 'ap_hi', 'type': 'continuous', 'description': 'Systolic blood pressure (mm Hg)', 'min': cardio_data['ap_hi'].min(), 'max': cardio_data['ap_hi'].max()},
        {'name': 'ap_lo', 'type': 'continuous', 'description': 'Diastolic blood pressure (mm Hg)', 'min': cardio_data['ap_lo'].min(), 'max': cardio_data['ap_lo'].max()},
        {'name': 'cholesterol', 'type': 'categorical', 'description': 'Cholesterol level (1: normal, 2: above normal, 3: well above normal)',
         'value': '1', 'label': 'Normal'},
        {'name': 'cholesterol', 'type': 'categorical', 'description': 'Cholesterol level', 'value': '2', 'label': 'Above Normal'},
        {'name': 'cholesterol', 'type': 'categorical', 'description': 'Cholesterol level', 'value': '3', 'label': 'Well Above Normal'},
        {'name': 'gluc', 'type': 'categorical', 'description': 'Glucose level (1: normal, 2: above normal, 3: well above normal)',
         'value': '1', 'label': 'Normal'},
        {'name': 'gluc', 'type': 'categorical', 'description': 'Glucose level', 'value': '2', 'label': 'Above Normal'},
        {'name': 'gluc', 'type': 'categorical', 'description': 'Glucose level', 'value': '3', 'label': 'Well Above Normal'},
        {'name': 'smoke', 'type': 'categorical', 'description': 'Smoking status (0 = non-smoker, 1 = smoker)', 'value': '0', 'label': 'Non-Smoker'},
        {'name': 'smoke', 'type': 'categorical', 'description': 'Smoking status', 'value': '1', 'label': 'Smoker'},
        {'name': 'alco', 'type': 'categorical', 'description': 'Alcohol intake (0 = no, 1 = yes)', 'value': '0', 'label': 'No'},
        {'name': 'alco', 'type': 'categorical', 'description': 'Alcohol intake', 'value': '1', 'label': 'Yes'},
        {'name': 'active', 'type': 'categorical', 'description': 'Physical activity (0 = no, 1 = yes)', 'value': '0', 'label': 'Inactive'},
        {'name': 'active', 'type': 'categorical', 'description': 'Physical activity', 'value': '1', 'label': 'Active'},
        {'name': 'cardio', 'type': 'categorical', 'description': 'Cardiovascular disease presence (0 = no, 1 = yes)', 'value': '0', 'label': 'No CVD'},
        {'name': 'cardio', 'type': 'categorical', 'description': 'Cardiovascular disease presence', 'value': '1', 'label': 'CVD'},
        # Overall dataset description.
        {'name': 'dataset', 'type': 'overall', 'description': 'Kaggle Cardiovascular Disease dataset',
         'overall_description': 'This dataset contains clinical features for cardiovascular disease risk prediction, including demographic, anthropometric, and blood pressure measurements.'}
    ]
    cardio_dict_df = pd.DataFrame(dict_entries)
    cardio_dict_csv = "cardio_dict.csv"
    cardio_dict_df.to_csv(cardio_dict_csv, index=False)
    logging.info("Cardiovascular Disease dataset and dictionary saved.")


def main():
    """
    Main function to download and prepare all clinical datasets for autoencoder training.
    """
    download_and_prepare_framingham()
    download_and_prepare_uci_heart()
    download_and_prepare_cardio()
    logging.info("All datasets have been downloaded and prepared.")


if __name__ == "__main__":
    main()
