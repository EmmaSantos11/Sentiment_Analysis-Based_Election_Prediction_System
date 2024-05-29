import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load the CSV files
try:
    trump_data = pd.read_csv('Trumpall2.csv')
    biden_data = pd.read_csv('Bidenall2.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise SystemExit("Please ensure the CSV files are in the correct directory.")

# Combine the data for both Trump and Biden
data = pd.concat([trump_data, biden_data])

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis and calculate compound polarity scores
data["Polarity"] = data["text"].apply(lambda x: sid.polarity_scores(x)['compound'])

# Assign sentiment labels based on compound polarity scores
data["Sentiment"] = np.where(data["Polarity"] > 0, "Positive", np.where(data["Polarity"] < 0, "Negative", "Neutral"))

# Filter out neutral sentiments for simplicity
data = data[data["Sentiment"] != "Neutral"]

# Map sentiment labels to numeric values
data["Sentiment"] = data["Sentiment"].map({"Positive": 1, "Negative": 0})

# Extract features and labels
X = data["text"]
y = data["Sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict the sentiment of new tweets
def predict_sentiment(tweet):
    tweet_tfidf = vectorizer.transform([tweet])
    prediction = model.predict(tweet_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

# Function to update sentiment counts and predict the likely winner
def update_sentiment_counts(new_tweet, candidate, positive_counts, negative_counts):
    sentiment = predict_sentiment(new_tweet)
    if candidate == "Trump":
        if sentiment == "Positive":
            positive_counts[0] += 1
        else:
            negative_counts[0] += 1
    elif candidate == "Biden":
        if sentiment == "Positive":
            positive_counts[1] += 1
        else:
            negative_counts[1] += 1
    
    # Predict the likely winner based on sentiment counts
    winner = "Trump" if positive_counts[0] - negative_counts[0] > positive_counts[1] - negative_counts[1] else "Biden"
    return sentiment, winner

# Initialize sentiment counts
total_positive_trump = (data[data["text"].str.contains('Trump', case=False)]["Sentiment"] == 1).sum()
total_negative_trump = (data[data["text"].str.contains('Trump', case=False)]["Sentiment"] == 0).sum()
total_positive_biden = (data[data["text"].str.contains('Biden', case=False)]["Sentiment"] == 1).sum()
total_negative_biden = (data[data["text"].str.contains('Biden', case=False)]["Sentiment"] == 0).sum()

positive_counts = [total_positive_trump, total_positive_biden]
negative_counts = [total_negative_trump, total_negative_biden]

# Function to accept new tweets interactively
def interactive_mode():
    while True:
        new_tweet = input("Enter a new tweet (or 'exit' to quit): ")
        if new_tweet.lower() == 'exit':
            break
        candidate = input("Enter the candidate (Trump/Biden): ")
        sentiment, winner = update_sentiment_counts(new_tweet, candidate, positive_counts, negative_counts)
        print(f"New tweet sentiment: {sentiment}")
        print(f"Updated Positive Counts: Trump - {positive_counts[0]}, Biden - {positive_counts[1]}")
        print(f"Updated Negative Counts: Trump - {negative_counts[0]}, Biden - {negative_counts[1]}")
        print(f"Likely Winner based on sentiment analysis: {winner}")
        
        # Plot the updated sentiment counts
        plot_sentiments()

# Function to plot the sentiments
def plot_sentiments():
    names = ["Trump", "Biden"]

    # Define the y values for positive and negative sentiments
    positive_values = [positive_counts[0], positive_counts[1]]
    negative_values = [negative_counts[0], negative_counts[1]]

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.bar(names, positive_values, color='green', label='Positive')
    plt.bar(names, negative_values, color='red', label='Negative', bottom=positive_values)
    plt.xlabel('Candidates')
    plt.ylabel('Number of Tweets')
    plt.title('Sentiment Analysis of Trump and Biden Tweets')
    plt.legend()
    plt.show()

# Run the interactive mode
interactive_mode()