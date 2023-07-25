# Movie-Sentiment-Analysis


![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)

This is a web application built with Flask that performs sentiment analysis on movie reviews. The application uses a machine learning model to predict whether a movie review is positive or negative.

## Table of Contents
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [License](#license)

## Getting Started

To run this application locally, follow the instructions in the [Installation](#installation) section.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/flask-movie-sentiment-analysis.git
   cd flask-movie-sentiment-analysis


Open your web browser and navigate to `http://localhost:5000` to access the application.

## How It Works

The application uses a machine learning model trained on a dataset of movie reviews to perform sentiment analysis. The model is a Random Forest classifier that uses TF-IDF vectorization to convert the review text into numerical features.

When a user enters a movie review, the application preprocesses the text, tokenizes it, and removes stopwords. Then, the TF-IDF vectorizer transforms the preprocessed text into numerical features. Finally, the model predicts the sentiment of the review and displays the result on the web page.

## Dataset

The movie reviews dataset used for training the machine learning model is available in the `movie_reviews.csv` file. The dataset contains two columns: `review` (text of the movie review) and `sentiment` (label indicating positive or negative sentiment).

## Technologies Used

- Python
- Flask
- NLTK (Natural Language Toolkit)
- Pandas
- Scikit-learn


![image](https://github.com/aadityazz/Movie-Sentiment-Analysis/assets/67819043/bf13f525-2ef9-4589-abb9-835df272e369)
![image](https://github.com/aadityazz/Movie-Sentiment-Analysis/assets/67819043/f48d1124-922b-4979-b185-e0b2f5a6d4c7)


