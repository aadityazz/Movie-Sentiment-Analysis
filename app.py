from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample


# Function for cleaning the review
def clean_review(review):
    ''' ... Your implementation for clean_review() ...

    Input:
        review: a string containing a movie review.
    Output:
        review_cleaned: a processed review.
    '''

    # 1. Remove extra spaces
    review = re.sub(" +", " ", review)

    # 2. Remove punctuations
    review = re.sub(r'[^\w\s]', '', review)

    # 3. Remove hyperlinks
    review = re.sub(r'http\S+', '', review)

    # 4. Remove special characters
    review = re.sub("[^A-Za-z0-9.]+", " ", review)

    # 5. Lowercasing
    review = review.lower()

    # 6. Tokenization
    tokens = word_tokenize(review)

    # 7. Removing stopwords
    stop_words = set(stopwords.words('english'))
    new_review = [i for i in tokens if not i in stop_words]

    # 8. Stemming
    ps = PorterStemmer()
    new_rev = [ps.stem(word) for word in new_review]

    # Creating sentence from the list of words
    review_cleaned = ' '.join(word for word in new_rev)

    return review_cleaned

# Function for predicting sentiment using Naive Bayes model
def naive_bayes_predict(review, logprior, loglikelihood):
    ''' ... Your implementation for naive_bayes_predict() ...
    Params:
        review: a string representing a movie review.
        logprior: the log prior computed from the training data.
        loglikelihood: a dictionary mapping each word to its log likelihood value.

    Return:
        sentiment: the predicted sentiment for the review (0 for positive, 1 for negative).
    '''

    # Process the review to get a list of words
    word_l = clean_review(review).split()

    # Initialize probability to zero
    total_prob = 0

    # Add the log prior
    total_prob += logprior

    for word in word_l:
        # Check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # Add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]

    # Predict sentiment based on the total probability
    sentiment = 0 if total_prob <= 0 else 1

    return sentiment

# Function for counting word occurrences in reviews
def review_counter(output_occurrence, reviews, positive_or_negative):
    ''' ... Your implementation for review_counter() ...

    Params:
        output_occurrence: a dictionary that will be used to map each word to its frequency.
        reviews: a list of movie reviews.
        positive_or_negative: a list corresponding to the sentiment of each review (either 0 or 1).

    Return:
        output_occurrence: a dictionary mapping each word to its frequency.
    '''
    for label, review in zip(positive_or_negative, reviews):
        # Split the review into cleaned words using the clean_review() function
        split_review = clean_review(review).split()

        for word in split_review:
            # Define the key as the word and label tuple
            key = (word, label)

            # If the key exists in the dictionary, increment the count
            if key in output_occurrence:
                output_occurrence[key] += 1
            # Else, if the key is new, add it to the dictionary and set the count to 1
            else:
                output_occurrence[key] = 1

    return output_occurrence

# Function for training Naive Bayes model
def train_naive_bayes(freqs, train_x, train_y):
    ''' ... Your implementation for train_naive_bayes() ...
     Input:
        freqs: dictionary from (word, label) to how often the word appears.
        train_x: a list of movie reviews.
        train_y: a list of labels corresponding to the reviews (0 for positive, 1 for negative).

    Output:
        logprior: the log prior probability (log of the ratio of negative and positive documents).
        loglikelihood: the log likelihood of the Naive Bayes model (log of the ratio of word frequencies for negative and positive classes).
    '''
    loglikelihood = {}
    logprior = 0

    # Calculate V, the number of unique words in the vocabulary
    vocab = set([key[0] for key in freqs.keys()])
    V = len(vocab)

    # Calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    for pair in freqs.keys():
        if pair[1] == 0:
            num_pos += freqs[pair]
        else:
            num_neg += freqs[pair]

    # Calculate num_doc, the number of documents
    num_doc = len(train_y)

    # Calculate D_pos, the number of positive documents
    pos_num_docs = len(train_y.loc[train_y == 0])

    # Calculate D_neg, the number of negative documents
    neg_num_docs = len(train_y.loc[train_y == 1])

    # Calculate logprior
    logprior = np.log(neg_num_docs) - np.log(pos_num_docs)

    # For each word in the vocabulary...
    for word in vocab:
        # Get the positive and negative frequency of the word
        freq_pos = freqs.get((word, 0), 0)
        freq_neg = freqs.get((word, 1), 0)

        # Calculate the probability that each word is positive and negative
        p_w_pos = (freq_pos + 1) / (num_pos + V)
        p_w_neg = (freq_neg + 1) / (num_neg + V)

        # Calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_neg / p_w_pos)

    return logprior, loglikelihood

# Function for testing Naive Bayes model
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    ''' ... Your implementation for test_naive_bayes() ...
    Input:
        test_x: A list of movie reviews for testing.
        test_y: The corresponding labels for the list of movie reviews.
        logprior: The log prior probability computed during training.
        loglikelihood: A dictionary with the log likelihoods for each word.

    Output:
        accuracy: The accuracy of the model (the fraction of correctly classified reviews).
        error: The error of the model (the fraction of misclassified reviews).
        results: The confusion matrix of the model.
        false_pos: A list of movie reviews misclassified as false positive.
        false_neg: A list of movie reviews misclassified as false negative.
    '''
    accuracy = 0
    y_hats = []

    for review in test_x:
        # If the prediction is > 0, the predicted class is 1 (negative sentiment)
        if naive_bayes_predict(review, logprior, loglikelihood) > 0:
            y_hat_i = 1
        # Otherwise, the predicted class is 0 (positive sentiment)
        else:
            y_hat_i = 0

        y_hats.append(y_hat_i)

    # Calculate the error (average of absolute differences between y_hats and test_y)
    error = np.mean(np.absolute(y_hats - test_y))

    # Calculate the accuracy
    accuracy = 1 - error

    # Rename actual and predicted values
    test_y = pd.Series(test_y, name='Actual')
    y_hats = pd.Series(y_hats, name='Predicted')

    # Create the confusion matrix
    results = confusion_matrix(test_y, y_hats)

    # Store false positive and false negative reviews
    false_pos = [review for review, pred in zip(test_x, y_hats) if pred == 1 and review not in test_x[test_y == 1]]
    false_neg = [review for review, pred in zip(test_x, y_hats) if pred == 0 and review not in test_x[test_y == 0]]

    return accuracy, error, results, false_pos, false_neg


# Load your movie reviews dataset (movie_reviews.csv)
df = pd.read_csv("movie_reviews.csv", sep=',', encoding='latin-1')

# Data Preprocessing and Model Training

# Perform upsampling to balance the classes
df_majority = df[df["sentiment"] == "positive"]
df_minority = df[df["sentiment"] == "negative"]

negative_upsample = resample(df_minority, replace=True, n_samples=df_majority.shape[0], random_state=101)

df_upsampled = pd.concat([df_majority, negative_upsample])
df_upsampled = df_upsampled.sample(frac=1)

negative_data_points_train = df_upsampled[df_upsampled["sentiment"] == "negative"].iloc[:10000]
positive_data_points_train = df_upsampled[df_upsampled["sentiment"] == "positive"].iloc[:10000]
negative_data_points_test = df_upsampled[df_upsampled["sentiment"] == "negative"].iloc[10000:]
positive_data_points_test = df_upsampled[df_upsampled["sentiment"] == "positive"].iloc[10000:]

X_train = pd.concat([negative_data_points_train["review"], positive_data_points_train["review"]])
y_train = pd.concat([negative_data_points_train["sentiment"], positive_data_points_train["sentiment"]])
X_test = pd.concat([negative_data_points_test["review"], positive_data_points_test["review"]])
y_test = pd.concat([negative_data_points_test["sentiment"], positive_data_points_test["sentiment"]])

output_map = {'positive': 0, 'negative': 1}
y_train = y_train.map(output_map)
y_test = y_test.map(output_map)

freqs = review_counter({}, X_train, y_train)
logprior, loglikelihood = train_naive_bayes(freqs, X_train, y_train)



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        # Preprocess the review
        cleaned_review = clean_review(review)

        # Perform sentiment analysis prediction
        sentiment = naive_bayes_predict(cleaned_review, logprior, loglikelihood)

        # Convert the sentiment result to a human-readable format
        if sentiment == 0:
            result = 'Positive'
        else:
            result = 'Negative'

        return render_template('result.html', review=review, result=result)

# Route to handle prediction requests via API
@app.route('/predict-endpoint', methods=['POST'])
def predict_endpoint():
    if request.method == 'POST':
        data = request.get_json()

        if 'review' not in data:
            return jsonify({'error': 'Please provide a "review" field in the request body.'}), 400

        review = data['review']

        # Clean the review
        cleaned_review = clean_review(review)

        # Predict sentiment using the Naive Bayes model
        sentiment = naive_bayes_predict(cleaned_review, logprior, loglikelihood)

        # Map the sentiment to a human-readable label
        sentiment_label = "Positive" if sentiment == 0 else "Negative"

        return jsonify({'review': review, 'result': sentiment_label}), 200



if __name__ == '__main__':
    # Run the Flask app on the local development server
    app.run(debug=True)