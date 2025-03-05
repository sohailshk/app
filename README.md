# Real-Time Anomaly Detection for Industrial Component Degradation

## Overview
This project implements a real-time anomaly detection system for monitoring industrial component degradation. The dataset used consists of one year of recorded machine data, sourced from the European research and innovation project IMPROVE. The main goal is to predict anomalies in component behavior using Machine Learning techniques and provide predictive maintenance insights.

## Features
- **Machine Learning-Based Anomaly Detection**: Utilizes an SVM (Support Vector Machine) model trained on curated and preprocessed data.
- **Real-Time Synthetic Data Generation**: Simulates real-time industrial data to test anomaly predictions.
- **Predictive Maintenance**: Calculates and uploads maintenance logs while forecasting the next maintenance date.
- **Feature-Based Anomaly Detection**: Allows users to manually input feature ranges and get an alert if the values exceed permissible limits.
- **Image-Based Anomaly Detection**: Users can upload product images, and the system will generate a mask highlighting the anomalous regions.
- **Interactive Streamlit Dashboard**: Provides a real-time interface for anomaly monitoring and decision-making.

---

## we have also created a website for client that is a nextjs based frontend website to attract clients below are some phots odf it we werent able to add the files here 
![image](https://github.com/user-attachments/assets/7874b4af-56ec-448a-8f9d-cd02ea33d962)
![image](https://github.com/user-attachments/assets/1c9a259c-8bcf-4b77-a5d5-572c10f97f5a)



## Dataset Information
### Context
The dataset comprises recorded machine data of a degrading component over a 12-month period. It originates from the Vega shrink-wrapper used in large-scale food and beverage production lines.

### Content
- The machine wraps and heat-seals products using a cutting assembly that requires precise maintenance.
- The dataset includes 519 CSV files, each capturing an 8-second machine operation with a 4ms time resolution (2048 time-samples per file).
- The file naming format: `MM-DDTHHMMSS_NUM_modeX.csv`, where:
  - `MM` represents the month (1-12, not a calendar month).
  - `DD` is the day.
  - `HHMMSS` represents the start time of recording.
  - `NUM` is the sample number.
  - `modeX` ranges from 1 to 8, representing different operational conditions.





## Methodology
### 1. **Dataset Curation & Preprocessing**
- Applied **SVM-based filtering** to remove outliers and clean the dataset.
- Standardized feature selection and scaling techniques for optimal performance.
- Saved the processed dataset and trained model in `.pkl` format using `joblib`.

### 2. **Anomaly Detection System**
- Implemented a **real-time synthetic data generator** to mimic real-time machine operations.
- Deployed the trained model to predict anomalies dynamically using **Streamlit**.

### 3. **Predictive Maintenance**
- Every detected anomaly is logged into a maintenance database.
- The system predicts the **next maintenance date** based on historical anomaly patterns.

### 4. **Feature-Based Anomaly Detection**
- Users can manually input a **range of permissible values** for specific features.
- The system checks if the real-time data falls within the allowed limits.

### 5. **Image-Based Anomaly Detection**
- Users upload product images.
- The model processes and generates an **anomaly mask** highlighting defect regions.

---

## Installation & Usage
### Prerequisites
- Python 3.x
- Required Libraries:
  ```bash
  pip install pandas numpy scikit-learn joblib streamlit opencv-python
  ```

### Running the Application
1. **Train the Model:**
   ```bash
   python train_model.py
   ```
2. **Start Real-Time Detection:**
   ```bash
   streamlit run app.py
   ```
3. **Upload Data for Feature-Based Detection:** Navigate to the input section in Streamlit UI and provide the feature range.
4. **Upload Image for Image-Based Detection:** Upload an image to see the highlighted anomaly mask.

---

## Acknowledgments
- This dataset originates from the European research and innovation project IMPROVE.
- The model was developed using **Support Vector Machine (SVM)** for anomaly detection.

This project serves as a robust solution for industrial predictive maintenance and real-time anomaly monitoring. 🚀

