# Sentiment_Analysis_project

# Sentiment Analyzer - Movie Reviews

This project is a Machine Learning and Natural Language Processing (NLP) based solution for sentiment analysis of movie reviews. The model is trained using the IMDB Movie Review dataset and predicts whether a review is positive or negative.

## Overview

The **Sentiment Analyzer** project performs sentiment classification on movie reviews. Given a text input (a review), the model predicts whether the sentiment is positive or negative. This project utilizes popular NLP and ML techniques, including text preprocessing, tokenization, vectorization, and machine learning classification.

### Technologies Used:
- **Python**
- **Pandas** for data manipulation
- **Numpy** for numerical operations
- **Scikit-learn** for machine learning algorithms
- **TensorFlow/Keras** for deep learning models (if used)
- **NLTK** for natural language processing tasks
- **Matplotlib** for visualizations

## Dataset

The model is trained on the **IMDB Movie Review dataset**, which contains 50,000 reviews labeled as positive or negative. This dataset is commonly used for binary sentiment classification.

- Dataset: [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## Installation

### 1. Clone the Repository

To get started, clone this repository to your local machine:
git clone https://github.com/RukeshAthikari/sentiment_analysis_project.git

 2. Create a Virtual Environment
It’s recommended to use a virtual environment for this project:

cd sentiment-analyzer
python -m venv venv
3. pip install -r requirements.txt

4. Preprocessing the Data
Before training the model, text data is preprocessed. This involves:

Removing stop words
Tokenization (splitting the text into words)
Lemmatization (reducing words to their base form)

5. Training the Model
To train the sentiment analysis model, run the following command:
python train_model.py
This will:
Load and preprocess the IMDB dataset
Split the data into training and test sets
Train a machine learning model (e.g., Logistic Regression, Naive Bayes, or a deep learning model)
Save the trained model to a file (e.g., sentiment_model.pkl)

6. Making Predictions
Once the model is trained, you can use it to predict the sentiment of a new movie review. To do so, run:
python predict_sentiment.py "Your movie review here."
This will output either "Positive" or "Negative" based on the sentiment of the provided review.

7. Example Usage (Web Interface)
You can also use the model in a web interface. To start the Flask web app (if integrated), run:
python app.py
This will start a local server at http://127.0.0.1:5000/ where you can input movie reviews and get the sentiment prediction.

