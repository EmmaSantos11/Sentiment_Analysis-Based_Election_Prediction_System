import os
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

csv_file = 'comments_data.csv'

# Ensure global variables for sentiment counts and comments are initialized
candidates = {}
comments_df = pd.DataFrame(columns=['candidate', 'comment', 'sentiment_score', 'sentiment'])

# Load Comments from CSV
def load_comments_from_csv():
    global comments_df, candidates
    if os.path.exists(csv_file):
        comments_df = pd.read_csv(csv_file)
        if 'candidate' in comments_df.columns and 'comment' in comments_df.columns:
            for candidate in comments_df['candidate'].unique():
                candidates[candidate] = list(comments_df[comments_df['candidate'] == candidate]['comment'])
        else:
            print("CSV file does not contain required columns 'candidate' and 'comment'.")
    else:
        print("No existing comments data found.")

# Count Sentiments in Comments
def count_sentiments(candidate_name):
    candidate_comments = comments_df[comments_df['candidate'] == candidate_name]
    positive_comments = len(candidate_comments[candidate_comments['sentiment'] == 'positive'])
    negative_comments = len(candidate_comments[candidate_comments['sentiment'] == 'negative'])
    neutral_comments = len(candidate_comments[candidate_comments['sentiment'] == 'neutral'])
    return positive_comments, negative_comments, neutral_comments

# Predict 
def predict_winner():
    candidate_names = comments_df['candidate'].unique()
    sentiments = {}
    for candidate in candidate_names:
        positive, negative, neutral = count_sentiments(candidate)
        sentiments[candidate] = {'positive': positive, 'negative': negative, 'neutral': neutral}

    if sentiments:
        winner = max(sentiments, key=lambda candidate: sentiments[candidate]['positive'])
        return winner, sentiments
    else:
        return None, sentiments

def calculate_accuracy(sentiments):
    accuracies = {}
    for candidate, sentiment in sentiments.items():
        total_comments = sum(sentiment.values())
        positive_comments = sentiment['positive']
        accuracy = (positive_comments / total_comments) * 100 if total_comments > 0 else 0
        accuracies[candidate] = accuracy
    return accuracies

def plot_sentiments_comparison(sentiments):
    data = []
    for candidate, sentiment in sentiments.items():
        data.append([candidate, 'positive', sentiment['positive']])
        data.append([candidate, 'negative', sentiment['negative']])
        
    sentiment_df = pd.DataFrame(data, columns=['candidate', 'sentiment', 'count'])
    
    sns.barplot(x='candidate', y='count', hue='sentiment', data=sentiment_df)
    plt.title('Sentiment Comparison Between Candidates (Positive and Negative only)')
    plt.show()

def predictive_analysis():
    load_comments_from_csv()
    if not comments_df.empty:
        winner, sentiments = predict_winner()
        if winner:
            accuracies = calculate_accuracy(sentiments)
            total_positive_comments = sum([sentiments[c]['positive'] for c in sentiments])
            total_negative_comments = sum([sentiments[c]['negative'] for c in sentiments])
            winner_accuracy = accuracies[winner]
            average_percentage = (winner_accuracy / sum(accuracies.values())) * 100 if sum(accuracies.values()) > 0 else 0

            print(f"The likely winner of the election is: {winner}")
            print(f"System Accuracy: {winner_accuracy:.2f}%")
            print(f"Total Positive Comments: {total_positive_comments}")
            print(f"Total Negative Comments: {total_negative_comments}")
            print(f"Average Percentage of Likely Winner: {average_percentage:.2f}%")
            print("Accuracy of both candidates:")
            for candidate, accuracy in accuracies.items():
                print(f"{candidate}: {accuracy:.2f}%")
            
            plot_sentiments_comparison(sentiments)
        else:
            print("No comments available to analyze.")
    else:
        print("No comments available to analyze.")

if __name__ == "__main__":
    predictive_analysis()
