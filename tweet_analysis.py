import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the CSV files
try:
    trump_data = pd.read_csv('Trumpall2.csv')
    biden_data = pd.read_csv('Bidenall2.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise SystemExit("Please ensure the CSV files are in the correct directory.")

trump_txt = TextBlob(trump_data["text"][100])

biden_txt = TextBlob(biden_data["text"][500])

def pol(text):
    return TextBlob(text).sentiment.polarity

# Load the Twitter data from the CSV file
trump_data["Polarity"] = trump_data["text"].apply(pol)
biden_data["Polarity"] = biden_data["text"].apply(pol)

# For Trump data
trump_data["sentiment"] = np.where(trump_data["Polarity"]>0, "Positive", "Negative")
trump_data.loc[trump_data["Polarity"] == 0, "sentiment"] = "Neutral"

# For Biden data
biden_data["sentiment"] = np.where(biden_data["Polarity"]>0, "Positive", "Negative")
biden_data.loc[biden_data["Polarity"] == 0, "sentiment"] = "Neutral"

# Check if 'tweet' column exists in both dataframes
if 'text' not in trump_data.columns:
    raise KeyError("The 'text' column is not present in Trumpall2.csv.")
if 'text' not in biden_data.columns:
    raise KeyError("The 'text' column is not present in Bidenall2.csv.")

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Filter tweets containing 'Trump' and 'Biden'
trump_tweets = trump_data[trump_data['text'].str.contains('Trump', case=False)]
biden_tweets = biden_data[biden_data['text'].str.contains('Biden', case=False)]

# Perform sentiment analysis
trump_sentiments = trump_tweets['text'].apply(lambda x: sid.polarity_scores(x))
biden_sentiments = biden_tweets['text'].apply(lambda x: sid.polarity_scores(x))

# Extract the compound polarity scores from the sentiment analysis results
trump_sentiments = trump_tweets['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
trump_total_score = sum(trump_sentiments)

biden_sentiments = biden_tweets['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
biden_total_score = sum(biden_sentiments)

# Assigning sentiment labels based on compound polarity scores using NumPy's where function
trump_sentiments = trump_tweets['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
trump_sentiments = np.where(trump_sentiments > 0, 'Positive', np.where(trump_sentiments < 0, 'Negative', 'Neutral'))

biden_sentiments = biden_tweets['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
biden_sentiments = np.where(biden_sentiments > 0, 'Positive', np.where(biden_sentiments < 0, 'Negative', 'Neutral'))

# Convert numpy array to pandas Series
trump_sentiments_series = pd.Series(trump_sentiments)
biden_sentiments_series = pd.Series(biden_sentiments)

# Group by sentiment and count occurrences
count_trump = trump_sentiments_series.groupby(trump_sentiments_series).count()
count_biden = biden_sentiments_series.groupby(biden_sentiments_series).count()

# Separate tweets with neutral polarity for Trump and Biden
trump_neutral = trump_data[trump_data["Polarity"]==0]
biden_neutral = biden_data[biden_data["Polarity"]==0]

# Remove tweets with neutral polarity from the datasets
trump_data.drop(trump_data[trump_data["Polarity"]==0].index, inplace=True)
biden_data.drop(biden_data[biden_data["Polarity"]==0].index, inplace=True)

# Perform sentiment analysis and calculate compound polarity scores
trump_data["Polarity"] = trump_data["text"].apply(lambda x: sid.polarity_scores(x)['compound'])
biden_data["Polarity"] = biden_data["text"].apply(lambda x: sid.polarity_scores(x)['compound'])

# Assign sentiment labels based on compound polarity scores
trump_data["Sentiment"] = np.where(trump_data["Polarity"] > 0, "Positive", np.where(trump_data["Polarity"] < 0, "Negative", "Neutral"))
biden_data["Sentiment"] = np.where(biden_data["Polarity"] > 0, "Positive", np.where(biden_data["Polarity"] < 0, "Negative", "Neutral"))

# Get the total positive and negative values for Trump and Biden
total_positive_trump = (trump_data["Sentiment"] == "Positive").sum()
total_negative_trump = (trump_data["Sentiment"] == "Negative").sum()

total_positive_biden = (biden_data["Sentiment"] == "Positive").sum()
total_negative_biden = (biden_data["Sentiment"] == "Negative").sum()

# Define the names for Trump and Biden
names = ["Trump", "Biden"]

# Define the y values for positive and negative sentiments
positive_values = [total_positive_trump, total_positive_biden]
negative_values = [total_negative_trump, total_negative_biden]

# Plot the graph
plt.figure(figsize=(10, 6))
plt.bar(names, positive_values, color='green', label='Positive')
plt.bar(names, negative_values, color='red', label='Negative', bottom=positive_values)
plt.xlabel('Candidates')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Analysis of Trump and Biden Tweets')
plt.legend()


text = " ".join(text for text in trump_data.text)
text = " ".join(filter(lambda x:x[0]!='@', text.split()))

import re

text = re.sub(r"http\S+", "", text)
WC=WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)

plt.figure(figsize=(15,10))
plt.imshow(WC, interpolation='bilinear')

text = " ".join(text for text in biden_data.text)
text = " ".join(filter(lambda x:x[0]!='@', text.split()))

import re

text = re.sub(r"http\S+", "", text)
WC=WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)

plt.figure(figsize=(15,10))
plt.imshow(WC, interpolation='bilinear')



# Print the results
print("Trump Data Columns:", trump_data.columns)
print("Biden Data Columns:", biden_data.columns)
print("Sentiment analysis result for Trump and Biden tweets: ")
print(trump_sentiments, biden_sentiments)
print(biden_data["text"][500], trump_data["text"][100])
print("biden:", biden_txt.sentiment) 
print("trump:", trump_txt.sentiment)
print(biden_neutral.shape, trump_neutral.shape)
print(trump_data.shape, biden_data.shape )
print(count_biden, count_trump)
print("there have {} word in all text".format(len(text)))
# Display the chart
plt.show()