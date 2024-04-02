## Hotel Review Sentiment Classification

### Overview
- This project aims to classify hotel reviews into positive, negative, or neutral sentiments using deep learning techniques. The model predicts the sentiment of each review based on the review text and associated rating.

### Dataset Information
- The dataset contains over 20,000 rows with review and rating columns. Each row represents a hotel review, with the associated rating ranging from 0 to 4.

### Problem Statement
- Accurately gauging customer sentiments from hotel reviews is essential for addressing feedback effectively. However, existing methods faced difficulty in precisely classifying reviews, impacting decision-making for hotel management.

### Solution
- We developed a deep learning model using GRU (Gated Recurrent Unit) architecture, achieving an accuracy of over 87% in sentiment prediction across ratings. The model enables precise classification of reviews into positive, negative, or neutral sentiments, facilitating informed decision-making for hotel management.

### Tech Stack
- TensorFlow
- NLTK
- GRU (Gated Recurrent Unit)
- DVC (Data Version Control)
- MLFlow-Dagshub (Experiment Tracking)
- Docker (Product Containerization )
- Airflow (Pipeline Orchestation )
- Github Actions CI/CD
- Azure Blob Storage
- Azure Web App
- Streamlit Cloud


MLFLOW_TRACKING_URI=https://dagshub.com/Meetpanchal58/Hotel-Review-Sentiment-Classification.mlflow
python src/pipeline/Complete_Pipeline.py

Deployment - https://hotel-review-sentiment-classification.streamlit.app/
