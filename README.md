# shipment_optimization_AWS
The shipment optimization model using machine learning, powered by AWS and Streamlit. It gets retrained through github push, and infers on new data online.

## Overview

This project uses historical Amazon delivery data to:
- Train a Gradient Boosting model to predict delivery time
- Recommend optimal delivery settings (Vehicle, Agent_Age, Agent_Rating)
- Provide a user-friendly interface via Streamlit
- Run training on GitHub Actions and store models on AWS S3

The works is done by:
- GitHub Actions: Automatically retrains model and encoder on push (CI/CD)
- S3: Stores dataset, trained model (`.pkl`), and encoder
- Streamlit Cloud: Hosts the user interface (frontend) and performs real-time model inference

Note: To minimize the cloud cost, we aren't using AWS Lambda or SageMekar

## Project Structure


shipment_optimization_AWS/
│
├── src/
│   ├── prepare_data.py          # Reads data for training and validation of the model from S3
│   ├── train_model.py           # Trains the model and saves it and the encoder to S3
│   └── __init__.py
│
├── app/
│   └── streamlit_app.py         # User-facing UI
│
├── .github/
│   └── workflows/
│       └── train_on_push.yml    # GitHub Actions workflow
│
├── data
│     └── amazon_delivery.csv    # This is just a backup of the data loaded on S3 and not directly used by the code
│
├── requirements.txt
└── README.md


## Features

- Predicts delivery time using GradientBoostingRegressor
- Optimizes delivery conditions for best timing
- Retrains automatically on `main` branch push
- Uses AWS S3 for model storage
- Deployed with Streamlit for user input and recommendations

## Security

- Sensitive info (AWS keys, email credentials) managed via **GitHub Secrets**
- `.streamlit/secrets.toml` uses injected secrets (not stored in repo)

## Notifications

You will get an email at YOUR_EMAIL (if you update the feature in GitHub Action) each time model retraining is complete.

## Setup

Dependencies will be installed by GitHub Actions in run.

