import os
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import streamlit as st
import tweepy

# Ensure initialization of session state variables
if 'comments_df' not in st.session_state:
    st.session_state['comments_df'] = pd.DataFrame(columns=['candidate', 'comment', 'sentiment_score', 'sentiment'])
    

if 'candidates' not in st.session_state:
    st.session_state['candidates'] = {}

if 'sentiments' not in st.session_state:
    st.session_state['sentiments'] = {}

csv_file = 'comments_data.csv'

# Function to add candidates
def add_candidates(candidate1, candidate2):
    st.session_state['candidates'][candidate1] = []
    st.session_state['candidates'][candidate2] = []
    st.success(f"Candidates '{candidate1}' and '{candidate2}' added.")

# Function to add a new comment
def add_comment(candidate_name, comment, sentiment_score, sentiment):
    try:
        # Ensure comments_df is initialized
        if 'comments_df' not in st.session_state:
            st.session_state['comments_df'] = pd.DataFrame(columns=['candidate', 'comment', 'sentiment_score', 'sentiment'])
        
        # Add a new row to the DataFrame
        st.session_state['comments_df'] = st.session_state['comments_df'].append({
            'candidate': candidate_name,
            'comment': comment,
            'sentiment_score': sentiment_score,
            'sentiment': sentiment
        }, ignore_index=True)
        
        st.success("Comment added successfully.")
    except Exception as e:
        st.error(f"Error adding comment: {e}")

# Function to analyze sentiment
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        sentiment = 'positive'
    elif sentiment_score == 0:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    return sentiment_score, sentiment

# Function to count sentiments for each candidate
def count_sentiments(candidate_name):
    candidate_comments = st.session_state['comments_df'][st.session_state['comments_df']['candidate'] == candidate_name]
    positive_comments = len(candidate_comments[candidate_comments['sentiment'] == 'positive'])
    negative_comments = len(candidate_comments[candidate_comments['sentiment'] == 'negative'])
    neutral_comments = len(candidate_comments[candidate_comments['sentiment'] == 'neutral'])
    return positive_comments, negative_comments, neutral_comments

# Function to plot sentiment distribution for each candidate
def plot_candidate_sentiment_distribution(candidate_name):
    candidate_comments = st.session_state['comments_df'][(st.session_state['comments_df']['candidate'] == candidate_name) & 
                                                        (st.session_state['comments_df']['sentiment'] != 'neutral')]
    sentiment_counts = candidate_comments['sentiment'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette={'positive': 'green', 'negative': 'red'})
    plt.title(f'Sentiment Distribution for {candidate_name} (Positive and Negative only)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)


# Function to plot overall sentiment distribution
def plot_sentiment_distribution():
    sentiment_counts = st.session_state['comments_df'][st.session_state['comments_df']['sentiment'] != 'neutral']['sentiment'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette={'positive': 'green', 'negative': 'red'})
    plt.title('Overall Sentiment Distribution (Positive and Negative only)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)

# Function to plot word cloud
def plot_wordcloud(candidate_name):
    text = " ".join(comment for comment in st.session_state['candidates'][candidate_name])
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {candidate_name}")
    st.pyplot(plt)

# Function to get top comments
def get_top_comments(candidate_name, sentiment, top_n=10):
    candidate_comments = st.session_state['comments_df'][(st.session_state['comments_df']['candidate'] == candidate_name) & 
                                                         (st.session_state['comments_df']['sentiment'] == sentiment)]
    candidate_comments.loc[:, 'sentiment_score'] = pd.to_numeric(candidate_comments['sentiment_score'], errors='coerce')
    candidate_comments = candidate_comments.dropna(subset=['sentiment_score'])
    return candidate_comments.nlargest(top_n, 'sentiment_score')

# Function to predict the likely winner
def predict_winner():
    candidate_names = st.session_state['comments_df']['candidate'].unique()
    sentiments = {}
    for candidate in candidate_names:
        positive, negative, neutral = count_sentiments(candidate)
        sentiments[candidate] = {'positive': positive, 'negative': negative, 'neutral': neutral}

    if sentiments:
        winner = max(sentiments, key=lambda candidate: sentiments[candidate]['positive'])
        return winner, sentiments
    else:
        return None, sentiments

# Function to calculate system accuracy
def calculate_accuracy(sentiments):
    accuracies = {}
    for candidate, sentiment in sentiments.items():
        total_comments = sum(sentiment.values())
        positive_comments = sentiment['positive']
        accuracy = (positive_comments / total_comments) * 100 if total_comments > 0 else 0
        accuracies[candidate] = accuracy
    return accuracies

# Function to plot sentiments comparison
def plot_sentiments_comparison(sentiments):
    data = []
    for candidate, sentiment in sentiments.items():
        data.append([candidate, 'positive', sentiment['positive']])
        data.append([candidate, 'negative', sentiment['negative']])
        
    sentiment_df = pd.DataFrame(data, columns=['candidate', 'sentiment', 'count'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='candidate', y='count', hue='sentiment', data=sentiment_df, palette={'positive': 'green', 'negative': 'red'})
    plt.title('Sentiment Comparison Between Candidates (Positive and Negative only)')
    plt.xlabel('Candidate')
    plt.ylabel('Count')
    st.pyplot(plt)

# Function to save a single comment to CSV
def save_comment_to_csv(candidate_name, comment, sentiment_score, sentiment):
    with open(csv_file, 'a') as f:
        f.write(f"{candidate_name},{comment},{sentiment_score},{sentiment}\n")

# Function to save data to CSV
def save_data_to_csv():
    st.session_state['comments_df'].to_csv(csv_file, index=False)
    st.success("Data saved to comments_data.csv")

# Function to clear all comments and reset
def clear_all_comments():
    st.session_state['comments_df'] = pd.DataFrame(columns=['candidate', 'comment', 'sentiment_score', 'sentiment'])
    st.session_state['candidates'] = {}
    st.session_state['sentiments'] = {}
    if os.path.exists(csv_file):
        os.remove(csv_file)
    st.success("All comments cleared and app reset.")

# Function to load comments from CSV
def load_comments_from_csv():
    try:
        global comments_df
        if os.path.exists(csv_file):
            st.session_state['comments_df'] = pd.read_csv(csv_file)
            if 'candidate' in st.session_state['comments_df'].columns and 'comment' in st.session_state['comments_df'].columns:
                for candidate in st.session_state['comments_df']['candidate'].unique():
                    st.session_state['candidates'][candidate] = list(st.session_state['comments_df'][st.session_state['comments_df']['candidate'] == candidate]['comment'])
                st.success("Comments loaded from comments_data.csv")
            else:
                st.warning("CSV file does not contain required columns 'candidate' and 'comment'.")
        else:
            st.warning("No existing comments data found.")
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {e}")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Function to get total number of comments
def total_comments():
    return len(st.session_state['comments_df'])

# Function to get the total number of positive, neutral, and negative comments
def total_positive_neutral_negative_comments():
    positive_comments = st.session_state['comments_df'][st.session_state['comments_df']['sentiment'] == 'positive']
    neutral_comments = st.session_state['comments_df'][st.session_state['comments_df']['sentiment'] == 'neutral']
    negative_comments = st.session_state['comments_df'][st.session_state['comments_df']['sentiment'] == 'negative']
    return len(positive_comments), len(neutral_comments), len(negative_comments)

# Function to get total word count in comments
def total_word_count():
    return st.session_state['comments_df']['comment'].apply(lambda x: len(x.split())).sum()

# Function to handle single file upload
def handle_single_file_upload(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if 'candidate' in df.columns and 'comment' in df.columns:
            for _, row in df.iterrows():
                add_comment(row['candidate'], row['comment'])
            st.success("Comments from the uploaded file have been added.")
        else:
            st.error("Uploaded file must contain 'candidate' and 'comment' columns.")
    except Exception as e:
                   st.error(f"Error loading file: {e}")

# Function to handle multiple file uploads
def handle_multiple_file_uploads(uploaded_files):
    try:
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            if 'candidate' in df.columns and 'comment' in df.columns:
                for _, row in df.iterrows():
                    add_comment(row['candidate'], row['comment'])
            else:
                st.error(f"Uploaded file '{uploaded_file.name}' must contain 'candidate' and 'comment' columns.")
        st.success("Comments from the uploaded files have been added.")
    except Exception as e:
        st.error(f"Error loading files: {e}")

# Function to load the uploaded CSV file and update session state
def load_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if 'candidate' in df.columns and 'comment' in df.columns:
            st.session_state['comments_df'] = df
            for candidate in df['candidate'].unique():
                st.session_state['candidates'][candidate] = list(df[df['candidate'] == candidate]['comment'])
        else:
            st.error("CSV file must contain 'candidate' and 'comment' columns.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Function to clear all comments and reset the app
def clear_all_comments():
    st.session_state['comments_df'] = pd.DataFrame(columns=['candidate', 'comment', 'sentiment_score', 'sentiment'])
    st.session_state['candidates'] = {}
    st.session_state['sentiments'] = {}
    if os.path.exists(csv_file):
        os.remove(csv_file)
    st.success("All comments cleared and app reset.")

# Function to get Twitter API client
def get_twitter_api():
    bearer_token = st.secrets["twitter"]["bearer_token"]
    client = tweepy.Client(bearer_token)
    return client

# Function to fetch tweets
def fetch_tweets(query, count=100):
    client = get_twitter_api()
    try:
        response = client.search_recent_tweets(query=query, max_results=count, tweet_fields=['created_at', 'text', 'public_metrics', 'source'])
        tweets_data = []
        for tweet in response.data:
            tweets_data.append({
                'Tweet': tweet.text,
                'Created At': tweet.created_at,
                'Retweets': tweet.public_metrics['retweet_count'],
                'Likes': tweet.public_metrics['like_count'],
                'Source': tweet.source
            })
        tweets_df = pd.DataFrame(tweets_data)
        return tweets_df
    except tweepy.TweepyException as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

# Function to perform sentiment analysis on tweets
def perform_sentiment_analysis(df):
    df['Sentiment'] = df['Tweet'].apply(lambda x: "Positive" if "good" in x else "Negative")  # Replace with actual sentiment analysis
    return df

# Function to display tweets and perform sentiment analysis
def display_tweets_and_sentiments():
    query = st.text_input("Enter the query for tweets")
    if query:
        tweets_df = fetch_tweets(query)
        if not tweets_df.empty:
            st.write("Fetched Tweets:")
            st.write(tweets_df)
            
            # Perform sentiment analysis (replace with actual sentiment analysis function)
            tweets_df = perform_sentiment_analysis(tweets_df)
            st.write("Sentiment Analysis:")
            st.write(tweets_df)

# Main function for Streamlit app
def main():
    st.title("Sentiment Analysis-Based Election Prediction System")

    # Load comments from CSV file
    load_comments_from_csv()

    # Add candidates section
    st.header("Add Candidates")
    candidate1 = st.text_input("Candidate 1 Name")
    candidate2 = st.text_input("Candidate 2 Name")
    if st.button("Add Candidates"):
        if candidate1 and candidate2:
            add_candidates(candidate1, candidate2)
        else:
            st.error("Please enter both candidate names.")

    # Upload comments CSV section
    st.header("Upload Comments CSV")
    uploaded_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) == 1:
            handle_single_file_upload(uploaded_files[0])
        elif len(uploaded_files) > 1:
            handle_multiple_file_uploads(uploaded_files)

    # Comment entry and analysis section
    if st.session_state['candidates']:
        st.header("Enter Comments for Candidates")
        candidate_name = st.selectbox("Select Candidate", list(st.session_state['candidates'].keys()))
        comment = st.text_area("Enter Comment")
        if st.button("Submit Comment"):
            add_comment(candidate_name, comment)

        if st.button("End Comment Collection"):
            save_data_to_csv()

        if st.button("Run Sentiment Analysis"):
            if not st.session_state['comments_df'].empty:
                st.header("Sentiment Analysis Results")
                st.header("Sentiment Distributions")
                candidate_names = st.session_state['comments_df']['candidate'].unique()
                for candidate in candidate_names:
                    st.subheader(f"Sentiment Distribution for {candidate}")
                    plot_candidate_sentiment_distribution(candidate)

                st.header("Overall Sentiment Distribution")
                plot_sentiment_distribution()
                
                for candidate in candidate_names:
                    st.subheader(f"Word Cloud for {candidate}")
                    plot_wordcloud(candidate)
                
                st.header("Top Comments")
                for candidate in candidate_names:
                    st.subheader(f"Top Positive Comments for {candidate}")
                    st.write(get_top_comments(candidate, 'positive'))
                    st.subheader(f"Top Negative Comments for {candidate}")
                    st.write(get_top_comments(candidate, 'negative'))

                total_positive, total_neutral, total_negative = total_positive_neutral_negative_comments()
                total_words = total_word_count()
                st.write(f"Total positive comments: {total_positive}")
                st.write(f"Total neutral comments: {total_neutral}")
                st.write(f"Total negative comments: {total_negative}")
                st.write(f"Total word count in comments: {total_words}")
                
                # Store sentiments in session state
                st.session_state['sentiments'] = {candidate: count_sentiments(candidate) for candidate in candidate_names}
            else:
                st.warning("No comments available to analyze.")

        if st.button("Perform Predictive Analysis"):
            if 'sentiments' in st.session_state and st.session_state['sentiments']:
                st.header("Predictive Analysis Results")
                winner, sentiments = predict_winner()
                if winner:
                    accuracies = calculate_accuracy(sentiments)
                    st.write("Accuracy for each candidate:")
                    for candidate, accuracy in accuracies.items():
                        st.write(f"{candidate}: {accuracy:.2f}%")
                    st.write(f"The likely winner of the election is: **{winner}**")
                    st.balloons()
                    st.write("Congratulations to the winner!")

                    st.header("Sentiment Distributions")
                    for candidate in sentiments.keys():
                        st.subheader(f"Sentiment Distribution for {candidate}")
                        plot_candidate_sentiment_distribution(candidate)

                    st.header("Sentiment Comparison")
                    plot_sentiments_comparison(sentiments)
                else:
                    st.warning("No comments available to analyze.")
            else:
                st.warning("Please run the sentiment analysis first.")

    # Button to clear all comments and reset the app
    if st.button("Clear All Comments and Reset"):
        clear_all_comments()

    # Twitter Sentiment Analysis section
    st.title("Twitter Sentiment Analysis")
    display_tweets_and_sentiments()

# Run the main function
if __name__ == "__main__":
    main()

