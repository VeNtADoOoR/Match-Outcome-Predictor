⚽ Leagues match outcome predictor ⚽

A full-stack, modular Machine Learning pipeline and interactive web application that predicts European football match outcomes (Home Win, Draw, Away Win) using Extreme Gradient Boosting (XGBoost).

🌟 Project Overview

In this project I used a custom-built ETL pipeline to engineer dynamic, time-series football features. It calculates mathematical advantages between teams and feeds them into an XGBoost classifier, which translates complex historical data into live, percentage-based probability predictions via a Streamlit web interface.

Key Features :

--> Modular ETL Architecture: Data ingestion, feature engineering, and model training are decoupled into clean, maintainable micro-scripts.

--> Dynamic Feature Engineering: Calculates rolling form, goal difference momentum, and rest-day advantages while strictly preventing data leakage.

--> Interactive UI: A user-friendly Streamlit frontend with built-in European League filtering.
