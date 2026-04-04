# Cardiovascular Disease Prediction App

A Machine Learning-powered web application built with Python and Streamlit. This app allows users to input standard clinical parameters and instantly assesses their risk of heart disease using a trained Random Forest Classifier.

## 🧠 About the App

This project bridges predictive machine learning with an interactive user interface. 
* **The Model:** Trained on the classic Cleveland Heart Disease dataset using a Random Forest algorithm to ensure high accuracy and robust predictions.
* **The Frontend:** Built entirely in Python using Streamlit, providing a clean, responsive, and intuitive medical form for users.

## 📊 How It Works

```mermaid
graph TD
    A[heart.csv Dataset] -->|Trained via Jupyter Notebook| B(Random Forest Model)
    B -->|Exported as| C[heart_disease_model.pkl]
    C -->|Loaded into| D{Streamlit Web App}
    E[User Inputs Clinical Data] --> D
    D -->|Predicts Risk| F((Prediction Results))