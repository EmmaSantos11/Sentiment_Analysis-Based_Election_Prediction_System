import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tweepy
import toml
from textblob import TextBlob

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize session state if not initialized
if 'comments_df' not in st.session_state:
    st.session_state['comments_df'] = pd.DataFrame(columns=['Candidate', 'Comment', 'Sentiment', 'Sentiment_Score'])

# Function to load comments from a CSV file
def load_comments_from_csv(uploaded_file):
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            if 'Comment' in new_data.columns and 'Candidate' in new_data.columns:
                for index, row in new_data.iterrows():
                    sentiment, sentiment_score = analyze_sentiment(row['Comment'])
                    add_comment(row['Candidate'], row['Comment'], sentiment, sentiment_score)
                st.success("CSV file loaded successfully.")
            else:
                st.warning("The uploaded CSV does not contain 'Candidate' or 'Comment' columns.")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")

# Function to save data to CSV
def save_data_to_csv():
    st.session_state['comments_df'].to_csv('comments.csv', index=False)
    st.success("Data saved to comments.csv")

# Function to add candidates
def add_candidates(candidate1, candidate2):
    if candidate1 and candidate2:
        st.session_state['candidates'] = {candidate1: 0, candidate2: 0}
        st.success("Candidates added successfully.")
    else:
        st.warning("Please enter names for both candidates.")

# Function to add comment
def add_comment(candidate, comment, sentiment, sentiment_score):
    new_comment_df = pd.DataFrame({'Candidate': [candidate],
                                   'Comment': [comment],
                                   'Sentiment': [sentiment],
                                   'Sentiment_Score': [sentiment_score]})
    st.session_state['comments_df'] = pd.concat([st.session_state['comments_df'], new_comment_df], ignore_index=True)
    st.success("Comment added successfully.")

# Total comments
def total_comments():
    return len(st.session_state['comments_df'])

# Total positive, neutral, negative comments
def total_positive_neutral_negative_comments():
    positive = st.session_state['comments_df'][st.session_state['comments_df']['Sentiment'] == 'Positive'].shape[0]
    neutral = st.session_state['comments_df'][st.session_state['comments_df']['Sentiment'] == 'Neutral'].shape[0]
    negative = st.session_state['comments_df'][st.session_state['comments_df']['Sentiment'] == 'Negative'].shape[0]
    return positive, neutral, negative

# Total word count
def total_word_count():
    if 'Comment' in st.session_state['comments_df'].columns:
        return st.session_state['comments_df']['Comment'].apply(lambda x: len(str(x).split())).sum()
    return 0

# Function to analyze sentiment
def analyze_sentiment(comment):
    vs = analyzer.polarity_scores(comment)
    if vs['compound'] >= 0.05:
        sentiment = "Positive"
    elif vs['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, vs['compound']

# Function to plot sentiment distribution
def plot_sentiment_distribution():
    if 'Sentiment' in st.session_state['comments_df'].columns:
        sentiments = st.session_state['comments_df']['Sentiment'].value_counts()
        if not sentiments.empty:
            sentiments.plot(kind='bar', color=['green', 'blue', 'red'])
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            st.pyplot(plt)
            plt.clf()
        else:
            st.warning("No sentiment data available.")
    else:
        st.warning("No sentiment data available.")

# Plot candidate sentiment distribution
def plot_candidate_sentiment_distribution(candidate_name):
    candidate_comments = st.session_state['comments_df'][st.session_state['comments_df']['Candidate'] == candidate_name]
    sentiments = candidate_comments['Sentiment'].value_counts()
    sentiments.plot(kind='bar', color=['green', 'blue', 'red'])
    plt.title(f'Sentiment Distribution for {candidate_name}')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Comments')
    st.pyplot(plt)
    plt.clf()

# Plot word cloud
def plot_wordcloud(candidate_name):
    candidate_comments = st.session_state['comments_df'][st.session_state['comments_df']['Candidate'] == candidate_name]
    text = ' '.join(candidate_comments['Comment'].values)
    wordcloud = WordCloud(background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.clf()

# Get top comments
def get_top_comments(candidate_name, sentiment, top_n):
    candidate_comments = st.session_state['comments_df'][st.session_state['comments_df']['Candidate'] == candidate_name]
    top_comments = candidate_comments[candidate_comments['Sentiment'] == sentiment].sort_values(by='Sentiment_Score', ascending=False).head(top_n)
    return top_comments

# Function to predict the winner based on sentiment
def predict_winner():
    sentiments = st.session_state['comments_df'].groupby('Candidate')['Sentiment_Score'].mean()
    if not sentiments.empty:
        winner = sentiments.idxmax()
        return winner, sentiments
    else:
        return None, None

# Function to calculate accuracy
def calculate_accuracy(sentiments_dict):
    accuracies = {}
    for candidate, sentiment_score in sentiments_dict.items():
        accuracies[candidate] = (sentiment_score + 1) / 2 * 100  # convert from -1 to 1 scale to 0 to 100 scale
    return accuracies

# Function to plot sentiment comparison
def plot_sentiments_comparison(sentiments):
    sentiments.plot(kind='bar', color=['green', 'blue'])
    plt.title('Sentiments Comparison')
    plt.xlabel('Candidate')
    plt.ylabel('Average Sentiment Score')
    st.pyplot(plt)
    plt.clf()

# Function to clear all comments
def clear_all_comments():
    st.session_state['comments_df'] = pd.DataFrame(columns=['Candidate', 'Comment', 'Sentiment', 'Sentiment_Score'])
    st.success("All comments have been cleared.")

# Function to display top comments
def display_top_comments():
    if 'Sentiment' in st.session_state['comments_df'].columns:
        top_positive = st.session_state['comments_df'].sort_values(by='Sentiment_Score', ascending=False).head(5)
        top_negative = st.session_state['comments_df'].sort_values(by='Sentiment_Score').head(5)
        st.subheader("Top Positive Comments:")
        st.table(top_positive[['Candidate', 'Comment', 'Sentiment', 'Sentiment_Score']])
        st.subheader("Top Negative Comments:")
        st.table(top_negative[['Candidate', 'Comment', 'Sentiment', 'Sentiment_Score']])
    else:
        st.warning("No sentiment data available.")

# Function to read keys from secrets.toml
def read_twitter_keys():
    config = toml.load('secrets.toml')
    return config['twitter']['api_key'], config['twitter']['api_secret_key'], config['twitter']['access_token'], config['twitter']['access_token_secret']

# Function to get tweets
def get_tweets(candidate_name, count=100):
    api_key, api_secret_key, access_token, access_token_secret = read_twitter_keys()

    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweets = []
    try:
        for tweet in tweepy.Cursor(api.search_tweets, q=candidate_name, lang='en', tweet_mode='extended').items(count):
            tweets.append(tweet.full_text)
    except Exception as e:
        print(f"Error fetching tweets: {e}")

    return tweets

# Fuction to print word cloud for fetched tweets
def plot_wordcloud(candidate_name):
    tweets = get_tweets(candidate_name)
    if not tweets:
        print("No tweets found for the candidate.")
        return

    text = " ".join(tweets)
    if not text.strip():
        print("No text found in tweets.")
        return

    wordcloud = WordCloud(background_color='white').generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Function to plut candidate sentiment distribution from tweets
def plot_candidate_sentiment_distribution(candidate):
    tweets = get_tweets(candidate)
    if not tweets:
        print("No tweets found for the candidate.")
        return

    # Perform sentiment analysis and prepare data
    sentiments = analyze_sentiments(tweets)  # Assuming you have a function for sentiment analysis

    if sentiments.empty:
        print("No sentiments data available.")
        return

    # Plotting
    sentiments.plot(kind='bar', color=['green', 'blue', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title(f'Sentiment Distribution for {candidate}')
    plt.show()

# Streamlit app function
def main():
    st.title("Sentiment Analysis-Based Election Prediction System")

    menu = ["Add Candidates", "Add Comment", "View Comments", "Sentiment Analysis", "Data Management", "Twitter Data"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Add Candidates":
        st.subheader("Add Candidates")
        candidate1 = st.text_input("Enter the first candidate's name:")
        candidate2 = st.text_input("Enter the second candidate's name:")
        if st.button("Add Candidates"):
            add_candidates(candidate1, candidate2)

    elif choice == "Add Comment":
        st.subheader("Add Comment")
        if 'candidates' in st.session_state:
            candidate = st.selectbox("Select Candidate", options=list(st.session_state['candidates'].keys()))
            comment = st.text_area("Enter Comment")
            if st.button("Add Comment"):
                sentiment, sentiment_score = analyze_sentiment(comment)
                add_comment(candidate, comment, sentiment, sentiment_score)
        else:
            st.warning("Please add candidates first.")

    elif choice == "View Comments":
        st.subheader("View Comments")
        if 'candidates' in st.session_state:
            for candidate in st.session_state['candidates']:
                st.subheader(f"Top Positive Comments for {candidate}:")
                top_positive_comments = get_top_comments(candidate, 'Positive', 5)
                st.table(top_positive_comments[['Candidate', 'Comment', 'Sentiment', 'Sentiment_Score']])

                st.subheader(f"Top Negative Comments for {candidate}:")
                top_negative_comments = get_top_comments(candidate, 'Negative', 5)
                st.table(top_negative_comments[['Candidate', 'Comment', 'Sentiment', 'Sentiment_Score']])
        else:
            st.warning("Please add candidates first.")

    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")

        if st.button("Run Sentiment Analysis"):
            plot_sentiment_distribution()
            display_top_comments()

            st.subheader("Word Cloud for Candidates")
            if 'candidates' in st.session_state:
                for candidate in st.session_state['candidates']:
                    st.write(f"Word Cloud for {candidate}")
                    plot_wordcloud(candidate)

            st.subheader("Overall Sentiment Statistics")
            positive, neutral, negative = total_positive_neutral_negative_comments()
            st.write(f"Positive Comments: {positive}")
            st.write(f"Neutral Comments: {neutral}")
            st.write(f"Negative Comments: {negative}")

            st.subheader("Total Negative and Positive Sentiment for Each Candidate")
            if 'candidates' in st.session_state:
                for candidate in st.session_state['candidates']:
                    plot_candidate_sentiment_distribution(candidate)

        if st.button("Predict Winner"):
            winner, sentiments = predict_winner()
            if winner:
                st.write(f"Predicted Winner: {winner}")
                plot_sentiments_comparison(sentiments)

                accuracies = calculate_accuracy(sentiments)
                st.write("Accuracy of predictions:")
                for candidate, accuracy in accuracies.items():
                    st.write(f"{candidate}: {accuracy:.2f}%")

                avg_percentage = {candidate: (score + 1) / 2 * 100 for candidate, score in sentiments.items()}
                st.write("Average Percentage of Sentiments:")
                for candidate, avg in avg_percentage.items():
                    st.write(f"{candidate}: {avg:.2f}%")
                
                st.balloons()
            else:
                st.warning("No sentiment data available to predict winner.")

    elif choice == "Data Management":
        st.subheader("Data Management")
        if st.button("Save Data to CSV"):
            save_data_to_csv()
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="upload_csv")
        if st.button("Load Comments from CSV"):
            load_comments_from_csv(uploaded_file)
        if st.button("Clear All Comments"):
            clear_all_comments()

    elif choice == "Twitter Data":
        st.subheader("Fetch Real-Time Twitter Data")
        if 'candidates' in st.session_state:
            candidate = st.selectbox("Select Candidate", options=list(st.session_state['candidates'].keys()))
            tweet_count = st.number_input("Enter number of tweets to fetch", min_value=1, max_value=500, value=100)
            if st.button("Fetch Tweets"):
                tweets = get_tweets(candidate, tweet_count)
                for tweet in tweets:
                    sentiment, sentiment_score = analyze_sentiment(tweet)
                    add_comment(candidate, tweet, sentiment, sentiment_score)
                st.success("Tweets fetched and analyzed successfully.")
        else:
            st.warning("Please add candidates first.")

    st.sidebar.header("Statistics")
    st.sidebar.write(f"Total Comments: {st.session_state['comments_df'].shape[0]}")
    if 'Sentiment' in st.session_state['comments_df'].columns:
        positive_comments = st.session_state['comments_df'][st.session_state['comments_df']['Sentiment'] == 'Positive'].shape[0]
        neutral_comments = st.session_state['comments_df'][st.session_state['comments_df']['Sentiment'] == 'Neutral'].shape[0]
        negative_comments = st.session_state['comments_df'][st.session_state['comments_df']['Sentiment'] == 'Negative'].shape[0]
    else:
        positive_comments = neutral_comments = negative_comments = 0
    st.sidebar.write(f"Positive Comments: {positive_comments}")
    st.sidebar.write(f"Neutral Comments: {neutral_comments}")
    st.sidebar.write(f"Negative Comments: {negative_comments}")

if __name__ == '__main__':
    main()
